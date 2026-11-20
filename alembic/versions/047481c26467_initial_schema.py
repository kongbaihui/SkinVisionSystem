"""initial schema

Revision ID: 047481c26467
Revises: 
Create Date: 2026-04-05 20:48:12.297038

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '047481c26467'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=20), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('last_login', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('analysis_tasks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('patient_age', sa.Integer(), nullable=True),
    sa.Column('patient_symptoms', sa.String(length=500), nullable=True),
    sa.Column('patient_duration', sa.String(length=100), nullable=True),
    sa.Column('pipeline_mode', sa.String(length=50), nullable=True),
    sa.Column('inference_time_ms', sa.Float(), nullable=True),
    sa.Column('area_ratio', sa.Float(), nullable=True),
    sa.Column('irregularity_index', sa.Float(), nullable=True),
    sa.Column('color_variance', sa.Float(), nullable=True),
    sa.Column('features_json', sa.JSON(), nullable=True),
    sa.Column('mask_path', sa.String(length=255), nullable=True),
    sa.Column('roi_path', sa.String(length=255), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('llm_report', sa.Text(), nullable=True),
    sa.Column('request_id', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_tasks_id'), 'analysis_tasks', ['id'], unique=False)
    op.create_index(op.f('ix_analysis_tasks_request_id'), 'analysis_tasks', ['request_id'], unique=True)
    op.create_index(op.f('ix_analysis_tasks_user_id'), 'analysis_tasks', ['user_id'], unique=False)
    op.create_table('audit_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('action', sa.String(length=100), nullable=True),
    sa.Column('request_id', sa.String(length=100), nullable=True),
    sa.Column('detail_json', sa.JSON(), nullable=True),
    sa.Column('ip_address', sa.String(length=45), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_logs_request_id'), 'audit_logs', ['request_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)



def downgrade() -> None:
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_request_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_id'), table_name='audit_logs')
    op.drop_table('audit_logs')
    op.drop_index(op.f('ix_analysis_tasks_user_id'), table_name='analysis_tasks')
    op.drop_index(op.f('ix_analysis_tasks_request_id'), table_name='analysis_tasks')
    op.drop_index(op.f('ix_analysis_tasks_id'), table_name='analysis_tasks')
    op.drop_table('analysis_tasks')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

