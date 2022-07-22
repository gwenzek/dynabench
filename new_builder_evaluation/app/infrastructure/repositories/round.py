# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from app.infrastructure.models.models import Round
from app.infrastructure.repositories.abstract import AbstractRepository


class RoundRepository(AbstractRepository):
    def __init__(self) -> None:
        super().__init__(Round)
