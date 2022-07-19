from app.infrastructure.repositories.abstract import AbstractRepository
from app.infrastructure.models.models import TaskProposal

class TaskProposalRepository(AbstractRepository):
    def __init__(self)-> None:
        super().__init__(TaskProposal)