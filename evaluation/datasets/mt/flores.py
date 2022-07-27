# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

import yaml
from datasets.common import BaseDataset

from utils import helpers
from utils.evaluator import Job


logger = logging.getLogger(__name__)

FLORES101_SMALL1_LANGS = ["eng", "est", "hrv", "hun", "mkd", "srp"]
FLORES101_SMALL2_LANGS = ["eng", "ind", "jav", "msa", "tam", "tgl"]
FLORES101_FULL_LANGS = (
    "afr,amh,ara,asm,ast,azj,bel,ben,bos,bul,cat,ceb,ces,ckb,cym,dan,deu,ell,"
    "eng,est,fas,fin,fra,ful,gle,glg,guj,hau,heb,hin,hrv,hun,hye,ibo,ind,isl,"
    "ita,jav,jpn,kam,kan,kat,kaz,kea,khm,kir,kor,lao,lav,lin,lit,ltz,lug,luo,"
    "mal,mar,mkd,mlt,mon,mri,msa,mya,npi,nld,nob,nso,nya,oci,orm,ory,pan,pol,"
    "por,pus,ron,rus,slk,slv,sna,snd,som,spa,srp,swe,swh,tam,tel,tgk,tgl,tha,"
    "tur,ukr,umb,urd,uzb,vie,wol,xho,yor,zho_simp,zho_trad,zul"
).split(",")


class Flores101Base(BaseDataset):
    def __init__(
        self,
        task_code: str,
        name: str,
        round_id: int,
        local_path: str,
        partition: str,
        languages: list,
        directions: list = None,
        shard_by_lang: bool = False,
    ):
        self.local_path = local_path
        self.partition = partition
        self.shard_by_lang = shard_by_lang
        self.languages = languages
        self.directions = directions or lang_matrix(languages)
        self.task_code = task_code
        task = helpers.dotdict(
            {
                "s3_bucket": "nllb",
                "task_code": task_code,
                "aws_region": "us-west-2",
            }
        )
        super().__init__(
            task_code=task_code,
            name=name,
            round_id=round_id,
            db_connection_avail=False,
            db_connection_not_avail_task_info=task,
        )

    def _get_data_s3_path(self, perturb_prefix=None):
        # filename has the .jsonl extension.
        # When the dataset is sharded we don't put the extension
        # because AWS will match files by prefix.
        if "flores200-african" in self.name:
            return helpers.get_data_s3_path("flores_african", self.filename, perturb_prefix)

        name = self.name + "-" if self.shard_by_lang else self.filename
        return helpers.get_data_s3_path(self.task.task_code, name, perturb_prefix)

    def dataset_available_on_s3(self, perturb_prefix=None) -> bool:
        if not self.shard_by_lang:
            return super().dataset_available_on_s3(perturb_prefix)
        basepath = self._get_data_s3_path()
        required_paths = {basepath + f"{lang}.jsonl" for lang in self.languages}
        response = self.s3_client.list_objects_v2(
            Bucket=self.task.s3_bucket, Prefix=basepath
        )
        available_paths = {obj["Key"] for obj in response.get("Contents", [])}
        missing_paths = required_paths - available_paths
        if missing_paths and missing_paths != required_paths:
            logging.warning(
                "Sharded dataset is missing some parts."
                f"Bucket {self.task.s3_bucket}/{basepath}. Missing parts: "
                + f"{missing_paths}"
            )

        return not missing_paths

    def get_batch_transform_config(
        self, sagemaker_client, endpoint_name, job_name, perturb_prefix=None
    ) -> dict:
        batch_transform_config = super().get_batch_transform_config(
            sagemaker_client, endpoint_name, job_name, perturb_prefix
        )
        batch_transform_config["TransformInput"]["ContentType"] = "application/json"
        batch_transform_config["TransformInput"]["SplitType"] = "Line"
        batch_transform_config["BatchStrategy"] = "MultiRecord"
        batch_transform_config["MaxPayloadInMB"] = 1
        return batch_transform_config

    def label_field_converter(self, example):
        return {
            **(super().label_field_converter(example)),
            "tags": ["-".join((example["sourceLanguage"], example["targetLanguage"]))],
        }

    def compute_job_metrics(self, job: Job) -> Tuple[dict, dict]:
        """Custom evaluation for full Flores track.

        The output files are splitted by source language, and we are only interested
        in per direction results.
        So we evaluate each direction independently.

        Note: this implementation doesn't support pertubation, but it could/should
        be added.
        """
        assert not job.perturb_prefix, "Flores tasks don't support pertubation"
        if not self.shard_by_lang:
            # the default algorithm computes a bleu over all sentences,
            # where we want an average bleu across all directions
            eval_metrics, _ = super().compute_job_metrics(job)
            metadata_json = json.loads(eval_metrics["metadata_json"])
            perf_by_tag: List[dict] = metadata_json["perf_by_tag"]
        else:
            perf_by_tag = []
            for src in self.languages:
                src_perfs = self.eval_src_lang(job, src)
                perf_by_tag.extend(src_perfs)

        perf_metric_type = yaml.load(self.task.config_yaml, yaml.SafeLoader)[
            "perf_metric"
        ]["type"]
        return compute_averages(perf_metric_type, perf_by_tag), {}

    def eval_src_lang(self, job: Job, src: str) -> dict:
        # Flores out file are correct .jsonl format,
        # so we don't need to call `parse_outfile_and_upload`
        start = time.time()
        predictions = helpers.parse_s3_outfile(
            self.s3_client,
            self.s3_path(
                f"predictions/{job.endpoint_name}/raw/"
                + f"{self.task.task_code}/{self.name}-{src}.jsonl.out"
            ),
        )
        targets = helpers.parse_s3_outfile(
            self.s3_client,
            self.s3_path(
                f"datasets/flores/{self.task.task_code}/{self.name}-{src}.jsonl"
            ),
        )
        duration = (time.time() - start) / 60
        logger.debug(f"downloaded {src}-xx predictions, took {duration:.1f} minutes")
        targets = [self.label_field_converter(target) for target in targets]
        # Reuse the base eval method,
        # but we are only interested in the per-directions results
        raw_src_perfs = self.eval(predictions, targets)
        src_perfs = json.loads(raw_src_perfs["metadata_json"])["perf_by_tag"]
        directions = [m["tag"] for m in src_perfs]
        expected_directions = self.directions
        assert sorted(directions) == sorted(expected_directions)

        duration = (time.time() - start) / 60
        logger.debug(f"evaluated {src}-xx directions, took {duration:.1f} minutes")
        return src_perfs


class Flores101FullDev(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_full",
            name="flores101-full-dev",
            round_id=1,
            local_path=local_path,
            partition="dev",
            languages=FLORES101_FULL_LANGS,
            shard_by_lang=True,
        )


class Flores101FullDevTest(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_full",
            name="flores101-full-devtest",
            round_id=1,
            local_path=local_path,
            partition="devtest",
            languages=FLORES101_FULL_LANGS,
            shard_by_lang=True,
        )


class Flores101FullTest(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_full",
            name="flores101-full-test",
            round_id=1,
            local_path=local_path,
            partition="test",
            languages=FLORES101_FULL_LANGS,
            shard_by_lang=True,
        )


class Flores101Small1Dev(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_small1",
            name="flores101-small1-dev",
            round_id=1,
            local_path=local_path,
            partition="dev",
            languages=FLORES101_SMALL1_LANGS,
        )


class Flores101Small1DevTest(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_small1",
            name="flores101-small1-devtest",
            round_id=1,
            local_path=local_path,
            partition="devtest",
            languages=FLORES101_SMALL1_LANGS,
        )


class Flores101Small1Test(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_small1",
            name="flores101-small1-test",
            round_id=1,
            local_path=local_path,
            partition="test",
            languages=FLORES101_SMALL1_LANGS,
        )


class Flores101Small2Dev(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_small2",
            name="flores101-small2-dev",
            round_id=1,
            local_path=local_path,
            partition="dev",
            languages=FLORES101_SMALL2_LANGS,
        )


class Flores101Small2DevTest(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_small2",
            name="flores101-small2-devtest",
            round_id=1,
            local_path=local_path,
            partition="devtest",
            languages=FLORES101_SMALL2_LANGS,
        )


class Flores101Small2Test(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores101")
        super().__init__(
            task_code="flores_small2",
            name="flores101-small2-test",
            round_id=1,
            local_path=local_path,
            partition="test",
            languages=FLORES101_SMALL2_LANGS,
        )

# Note for Fula we explictly use Nigeria Fulfulde: fuv
FLORES200_AFRICAN_LANGS = (
    "afr,lin,ssw,amh,lug,tsn,nya,luo,umb,ian,fuv,nso,wol,"
    "hau,orm,xho,ibo,sna,tso,kam,som,yor,kin,swh,zul"
).split(",")
FLORES200_AFRICAN_WITH_ENG = (
    "afr,amh,nya,fuv,hau,ibo,kam,kin,lug,luo,nso,"
    "orm,sna,som,swh,ssw,tsn,umb,xho,tso,yor,zul"
).split(",")
FLORES200_AFRICAN_WITH_FRA = "kin,lin,swh,wol".split(",")
FLORES200_AFRICAN_SOUTH = [
    ("xho", "zul"),
    ("zul", "sna"),
    ("sna", "afr"),
    ("afr", "ssw"),
    ("ssw", "tsn"),
    ("tsn", "tso"),
    ("tso", "nso"),
    ("nso", "xho"),
]
FLORES200_AFRICAN_HORN = [
    ("swh", "amh"),
    ("amh","swh"),
    ("luo", "orm"), # Luo -> Oromo
    ("som", "amh"),
    ("orm", "som"),
    ("swh", "luo"),
    ("amh", "luo"),
    ("luo", "som"),
]
FLORES200_AFRICAN_NIGERIA = [
    ("hau", "ibo"),
    ("ibo", "yor"),
    ("yor", "fuv"),
    ("fuv", "hau"),
    ("ibo", "hau"),
    ("yor", "ibo"),
    ("hau", "fuv"),
    ("wol", "hau"),
    ("hau", "wol"),
    ("fuv", "wol"),
    ("wol", "fuv"),
]
FLORES200_AFRICAN_CENTRAL = [
    ("kin", "swh"),
    ("lug", "lin"),
    ("nya", "kin"), # Chichewa -> Kinyarwanda
    ("swh", "lug"),
    ("lin", "nya"),
    ("lin", "kin"),
    ("kin", "lug"),
    ("nya", "swh"),
]
FLORES200_AFRICAN_PANAFRICA = [
    ("amh", "zul"),
    ("yor", "swh"),
    ("swh", "yor"),
    ("zul", "amh"),
    ("kin", "hau"),
    ("hau", "kin"),
    ("nya", "som"),
    ("som", "nya"),
    ("xho", "lug"),
    ("lug", "xho"),
    ("wol", "swh"),
    ("swh", "wol"),
]

FLORES200_AFRICAN_DIRECTIONS = (
    [("eng", l) for l in FLORES200_AFRICAN_WITH_ENG]
    + [(l, "eng") for l in FLORES200_AFRICAN_WITH_ENG]
    + [("fra", l) for l in FLORES200_AFRICAN_WITH_FRA]
    + [(l, "fra") for l in FLORES200_AFRICAN_WITH_FRA]
    + FLORES200_AFRICAN_SOUTH
    + FLORES200_AFRICAN_HORN
    + FLORES200_AFRICAN_NIGERIA
    + FLORES200_AFRICAN_CENTRAL
    + FLORES200_AFRICAN_PANAFRICA
)


class Flores200AfricanDev(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores200")
        super().__init__(
            task_code="flores_small1",
            name="flores200-african-dev",
            round_id=1,
            local_path=local_path,
            partition="dev",
            languages=FLORES200_AFRICAN_LANGS,
            directions=FLORES200_AFRICAN_DIRECTIONS,
        )


class Flores200AfricanDevTest(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores200")
        super().__init__(
            task_code="flores_small1",
            name="flores200-african-devtest",
            round_id=1,
            local_path=local_path,
            partition="devtest",
            languages=FLORES200_AFRICAN_LANGS,
            directions=FLORES200_AFRICAN_DIRECTIONS,
        )


class Flores200AfricanTest(Flores101Base):
    def __init__(self):
        rootpath = os.path.dirname(sys.path[0])
        local_path = os.path.join(rootpath, "evaluation/data", "mt/flores200")
        super().__init__(
            task_code="flores_small1",
            name="flores200-african-test",
            round_id=1,
            local_path=local_path,
            partition="test",
            languages=FLORES200_AFRICAN_LANGS,
            directions=FLORES200_AFRICAN_DIRECTIONS,
        )


def lang_matrix(langs: List[str]) -> List[Tuple[str, str]]:
    directions = []
    for src, tgt in itertools.product(langs, langs):
        if src >= tgt:
            continue
        directions.append((src, tgt))
        directions.append((tgt, src))


@functools.lru_cache(maxsize=256)
def read_raw_data(folder: Path, split: str, lang: str) -> List[str]:
    """Makes sure we are reading each file at most once.

    There is 1000 sentences per file, one file per lang so the memory footprint is ok.
    """
    file = folder / f"{lang}.{split}"
    assert file.exists(), file
    return file.read_text().splitlines()


def write_json(sample: Dict[str, str], o: TextIO) -> None:
    print(json.dumps(sample, ensure_ascii=False), file=o)


def output_file(task_code: str, partition: str, outdir: Path, lang: str = None) -> Path:
    filename = f"flores101-{task_code}-{partition}"
    if lang:
        filename += f"-{lang}"
    return outdir / f"{filename}.jsonl"


def prepare_dataset(name: str = "Flores200AfricanDev", local_path: Path = None, outdir: Path = None):
    assert "Flores" in name
    flores = globals()[name]()
    local_path = local_path or Path(flores.local_path)
    assert local_path.exists(), f"Folder for {flores} not found: {local_path}"
    outdir = outdir or Path("/tmp") / "flores_json"

    return prepare(
        local_path,
        outdir,
        task_code=flores.task_code,
        directions=flores.directions,
        partition=flores.partition,
        s3_bucket="s3://nllb",
        shard=flores.shard_by_lang,
    )


def prepare(
    folder: Path,
    outdir: Path,
    task_code: str,
    directions: List[Tuple[str, str]],
    partition: str,
    s3_bucket: str,
    shard: bool = False,
):
    assert folder.exists()
    outdir.mkdir(exist_ok=True)

    if shard:
        writers: Dict[str, Tuple[Path, TextIO]] = {}
    else:
        # We only have one file
        outfile = output_file(task_code, partition, outdir)
        shared_writer = open(outfile, "w", encoding="utf-8")

    def _get_writer(lang: str) -> TextIO:
        if not shard:
            return shared_writer
        if lang not in writers:
            outfile = output_file(task_code, partition, outdir, lang)
            writers[lang] = (outfile, open(outfile, "w", encoding="utf-8"))

        outfile, writer = writers[lang]
        return writer

    lines = 0
    for src, tgt in directions:
        src_lines = read_raw_data(folder, partition, src)
        tgt_lines = read_raw_data(folder, partition, tgt)

        o_src = _get_writer(src)
        for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
            src_tgt_id_suffix = f"-{src}-{tgt}-{partition}"
            src_tgt = {
                "uid": str(i) + src_tgt_id_suffix,
                "sourceLanguage": src,
                "targetLanguage": tgt,
                "sourceText": src_line,
                "targetText": tgt_line,
            }
            write_json(src_tgt, o_src)
            lines += 1

        o_tgt = _get_writer(tgt)
        for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
            tgt_src_id_suffix = f"-{tgt}-{src}-{partition}"

            tgt_src = {
                "uid": str(i) + tgt_src_id_suffix,
                "sourceLanguage": tgt,
                "targetLanguage": src,
                "sourceText": tgt_line,
                "targetText": src_line,
            }
            write_json(tgt_src, o_tgt)
            lines += 1

    # Free memory
    read_raw_data.cache_clear()
    if shard:
        files = len(writers)
        logger.info(f"Wrote dataset. {lines:_d} lines, {files:_d} files.")
        for outfile, o in writers.values():
            o.close()
            _upload_file(task_code, partition, outfile, s3_bucket)
    else:
        logger.info(
            f"Wrote dataset {outfile}. {lines:_d} lines. Total size: "
            + f"{outfile.stat().st_size / 1024 / 1024:.1f}Mb"
        )
        shared_writer.close()
        _upload_file(task_code, partition, outfile, s3_bucket)


def _upload_file(task_code: str, partition: str, outfile: Path, s3_bucket: str) -> None:
    s3_path = "/".join([s3_bucket, "flores", f"flores_{task_code}", outfile.name])
    logger.info(f"Copying {outfile} to {s3_path}")
    cmd = ["aws", "s3", "cp", str(outfile), s3_path]
    logger.info(" ".join(cmd))
    logger.info(subprocess.check_output(cmd, text=True))


def compute_averages(perf_metric: str, perf_by_tag: List[dict]) -> dict:
    """
    Computes the average metrics from the per direction metrics.

    The output format match the one from vanilla `eval` method,
    and is ready to be put into the sql database.
    """
    avg_metrics = {}
    metric_names = {key for perf in perf_by_tag for key in perf["perf_dict"]}
    for key in metric_names:
        scores = [perf["perf_dict"][key] for perf in perf_by_tag]
        avg_metrics[key] = sum(scores) / len(scores)

    perf = avg_metrics[perf_metric]
    score_obj = {
        "perf": perf,
        "pretty_perf": f"{perf:.2f}",  # this is a BLEU score
        "metadata_json": json.dumps({**avg_metrics, "perf_by_tag": perf_by_tag}),
    }
    return score_obj


FLORES_DATASETS = {
    "flores101-full-dev": Flores101FullDev,
    "flores101-full-devtest": Flores101FullDevTest,
    "flores101-full-test": Flores101FullTest,
    "flores101-small1-dev": Flores101Small1Dev,
    "flores101-small1-devtest": Flores101Small1DevTest,
    "flores101-small1-test": Flores101Small1Test,
    "flores101-small2-dev": Flores101Small2Dev,
    "flores101-small2-devtest": Flores101Small2DevTest,
    "flores101-small2-test": Flores101Small2Test,
    "flores200-african-dev": Flores200AfricanDev,
    "flores200-african-devtest": Flores200AfricanDevTest,
    "flores200-african-test": Flores200AfricanTest,
}

if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(prepare_dataset)
