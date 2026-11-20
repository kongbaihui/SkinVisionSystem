"""add_patient_id_to_analysis_tasks

Revision ID: fa4e9cbd14e8
Revises: 047481c26467
Create Date: 2026-04-09 22:16:18.935046

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fa4e9cbd14e8'
down_revision: Union[str, None] = '047481c26467'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('analysis_tasks', sa.Column('patient_id', sa.String(length=30), nullable=True))



def downgrade() -> None:
    op.drop_column('analysis_tasks', 'patient_id')

