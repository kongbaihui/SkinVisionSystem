"""Keep the submitted image and version the clinician's report independently."""
from alembic import op
import sqlalchemy as sa

revision = 'e85c3198b479'
down_revision = 'd491b8f2a601'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('analysis_tasks', sa.Column('edited_report', sa.Text()))
    op.add_column('analysis_tasks', sa.Column('report_version', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('analysis_tasks', sa.Column('report_edited_at', sa.DateTime()))
    op.add_column('analysis_tasks', sa.Column('original_image', sa.LargeBinary()))
    op.add_column('analysis_tasks', sa.Column('original_media_type', sa.String(50)))


def downgrade():
    for name in ('original_media_type', 'original_image', 'report_edited_at', 'report_version', 'edited_report'):
        op.drop_column('analysis_tasks', name)
