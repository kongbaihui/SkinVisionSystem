"""add_role_to_users

Revision ID: c7d2f4e1b839
Revises: 3b9f1e2c8d05
Create Date: 2026-04-10 10:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7d2f4e1b839'
down_revision: Union[str, None] = '3b9f1e2c8d05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('role', sa.String(length=20), nullable=True, server_default='doctor'))


def downgrade() -> None:
    op.drop_column('users', 'role')
