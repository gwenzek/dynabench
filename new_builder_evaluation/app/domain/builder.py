# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import logging
import os
import shutil
import time
from zipfile import ZipFile

import boto3
import docker

from app import utils


log = logging.getLogger(__name__)


def get_model_name(model_zip_uri: str) -> str:
    model_name = model_zip_uri.split("/")[-1]
    model_name = "-".join(model_name.split(".")[0].replace(" ", "").split("-")[1:])
    assert model_name, f"Couldn't extract a proper model name from {model_zip_uri}"
    return model_name


class Builder:
    def __init__(self):
        self.session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ["AWS_REGION"],
        )
        self.s3 = self.session.client("s3")
        self.ecs = self.session.client("ecs")
        self.ec2 = self.session.client("ec2")
        self.lamda_ = self.session.client("lambda")
        self.api_gateway = self.session.client("apigateway")
        self.ecr = self.session.client("ecr")
        self.decentralized = bool(os.getenv("DYNABENCH_API"))
        # Required keys
        self.task_execution_role = os.environ["AWS_TASK_EXECUTION_ROLE"]
        self.s3_bucket = os.environ["AWS_S3_BUCKET"]
        self.docker_client = docker.from_env()

    def download_and_unzip(self, bucket_name: str, model_zip_uri: str) -> str:
        zip_name = model_zip_uri.split("/")[-1]
        folder_name = zip_name.split(".")[0]
        local_zip_file = f"./app/model/{zip_name}"
        if self.decentralized:
            # return utils.api_download_model(model, model.secret)
            shutil.copyfile(
                f"/home/ubuntu/submissions/flores_small1-dummy.zip",
                local_zip_file,
            )
        else:
            self.s3.download_file(bucket_name, model_zip_uri, local_zip_file)

        model_dir = f"./app/{folder_name}/model"
        with ZipFile(local_zip_file, "r") as zipObj:
            zipObj.extractall(model_dir)
        os.remove(local_zip_file)
        return model_dir

    def principal(self):
        zip_name, model_name = self.download_zip(
            "dynabench-api-ciro", "models/sentiment/1675-dynalab-base-sentiment.zip"
        )
        return self.unzip_file(zip_name), model_name

    def extract_ecr_configuration(self) -> dict:
        ecr_credentials = self.ecr.get_authorization_token()["authorizationData"][0]
        ecr_password = (
            base64.b64decode(ecr_credentials["authorizationToken"])
            .replace(b"AWS:", b"")
            .decode("utf-8")
        )
        ecr_url = ecr_credentials["proxyEndpoint"]
        return {"ecr_username": "AWS", "ecr_password": ecr_password, "ecr_url": ecr_url}

    def create_repository(self, repo_name: str) -> str:
        try:
            response = self.ecr.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={"scanOnPush": True},
            )
            return response["repository"]["repositoryUri"]
        except self.ecr.exceptions.RepositoryAlreadyExistsException as e:
            log.info(f"reusing repository '{repo_name}', because {e}")
            return repo_name

    def push_image_to_ECR(
        self, repository_name: str, folder_name: str, tag: str
    ) -> str:
        ecr_config = self.extract_ecr_configuration()
        self.docker_client.login(
            username=ecr_config["ecr_username"],
            password=ecr_config["ecr_password"],
            registry=ecr_config["ecr_url"],
        )
        image, _ = self.docker_client.images.build(path=folder_name, tag=tag)
        image.tag(repository=repository_name, tag=tag)
        self.docker_client.images.push(
            repository=repository_name,
            tag=tag,
            auth_config={
                "username": ecr_config["ecr_username"],
                "password": ecr_config["ecr_password"],
            },
        )
        log.info(f"Pushed docker image {image} to ECR {repository_name}")
        return f"{repository_name}:{tag}"

    def create_task_definition(self, name_task: str, repo: str) -> str:
        task_definition = self.ecs.register_task_definition(
            containerDefinitions=[
                {
                    "name": name_task,
                    "image": repo,
                }
            ],
            executionRoleArn=self.task_execution_role,
            family=name_task,
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            # TODO: This seems a lot. Also it should be configurable per task organizer
            cpu="4096",
            memory="20480",
        )
        return task_definition["taskDefinition"]["containerDefinitions"][0]["name"]

    def create_ecs_endpoint(self, name_task: str, repo: str) -> str:
        task_definition = self.create_task_definition(name_task, repo)
        network_conf = {
            "awsvpcConfiguration": {
                "subnets": [
                    os.getenv("SUBNET_1"),
                    os.getenv("SUBNET_2"),
                ],
                "assignPublicIp": "ENABLED",
                "securityGroups": [os.getenv("SECURITY_GROUP")],
            }
        }
        run_service = self.ecs.create_service(
            cluster=os.getenv("CLUSTER_TASK_EVALUATION", "heavy-task-evaluation"),
            serviceName=name_task,
            taskDefinition=task_definition,
            desiredCount=1,
            networkConfiguration=network_conf
            launchType="FARGATE",
        )

        while True:
            describe_service = self.ecs.describe_services(
                cluster=os.getenv("CLUSTER_TASK_EVALUATION"),
                services=[run_service["service"]["serviceArn"]],
            )
            service_state = describe_service["services"][0]["deployments"][0][
                "rolloutState"
            ]
            if service_state != "COMPLETED":
                time.sleep(60)
            # TODO handle other states
            else:
                arn_service = describe_service["services"][0]["serviceArn"]
                run_task = self.ecs.list_tasks(
                    cluster=os.getenv("CLUSTER_TASK_EVALUATION"), serviceName=name_task
                )["taskArns"]
                describe_task = self.ecs.describe_tasks(
                    cluster=os.getenv("CLUSTER_TASK_EVALUATION"), tasks=run_task
                )
                eni = self.eni.NetworkInterface(
                    describe_task["tasks"][0]["attachments"][0]["details"][1]["value"]
                )
                ip = eni.association_attribute["PublicIp"]
                break
        return ip, arn_service

    def delete_ecs_service(self, arn_service: str):
        self.ecs.delete_service(
            cluster=os.getenv("CLUSTER_TASK_EVALUATION"),
            service=arn_service,
            force=True,
        )

    def get_ip_ecs_task(self, model: str):
        zip_name, model_name = self.download_zip(os.getenv("AWS_S3_BUCKET"), model)
        folder_name = self.unzip_file(zip_name)
        repo = self.create_repository(model_name)
        tag = "latest"
        self.push_image_to_ECR(repo, f"./app/models/{folder_name}", tag)
        ip, arn_service = self.create_ecs_endpoint(model_name, f"{repo}")
        return ip, model_name, folder_name, arn_service

    def light_model_deployment(self, function_name: str, image_uri: str, role: str):
        lambda_function = self.lamda_.create_function(
            {
                "FunctionName": function_name,
                "Role": role,
                "Code": {"ImageUri": image_uri},
                "PackageType": "Image",
            }
        )
        return lambda_function
