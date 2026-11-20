"""Add nullable independent AI results; leave historical records unevaluated."""
from alembic import op
import sqlalchemy as sa

revision = 'd491b8f2a601'
down_revision = 'c7d2f4e1b839'
branch_labels = None
depends_on = None


def upgrade():
    for name in ['classification_json', 'reliability_json', 'model_info_json', 'report_metadata_json']:
        op.add_column('analysis_tasks', sa.Column(name, sa.JSON(), nullable=True))
    op.add_column('analysis_tasks', sa.Column('report_status', sa.String(50), nullable=True))
    for name in ['probability_path', 'overlay_path']:
        op.add_column('analysis_tasks', sa.Column(name, sa.String(255), nullable=True))


def downgrade():
    for name in ['classification_json', 'reliability_json', 'model_info_json', 'report_metadata_json',
                 'report_status', 'probability_path', 'overlay_path']:
        op.drop_column('analysis_tasks', name)
