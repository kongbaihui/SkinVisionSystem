"""add_gradcam_path_to_analysis_tasks

Revision ID: 3b9f1e2c8d05
Revises: fa4e9cbd14e8
Create Date: 2026-04-10 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3b9f1e2c8d05'
down_revision: Union[str, None] = 'fa4e9cbd14e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('analysis_tasks', sa.Column('gradcam_path', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('analysis_tasks', 'gradcam_path')
