import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def test_new_results_migration_preserves_old_records(tmp_path):
    database = tmp_path / 'legacy.db'
    env = {**os.environ, 'SKINVISION_DB_URL': 'sqlite:///' + database.as_posix()}
    def upgrade(revision):
        run = subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', revision],
                             env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
    upgrade('c7d2f4e1b839')
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO analysis_tasks (id,status,llm_report,features_json) VALUES (77,'completed','original report','{\"area_ratio\":3}')")
    upgrade('head')
    with sqlite3.connect(database) as connection:
        row = connection.execute('SELECT id,llm_report,classification_json,reliability_json,model_info_json,report_status,report_metadata_json,probability_path,overlay_path FROM analysis_tasks WHERE id=77').fetchone()
    assert row == (77,'original report',None,None,None,None,None,None,None)
    with sqlite3.connect(database) as connection:
        review = connection.execute('SELECT original_image,original_media_type,edited_report,report_version,report_edited_at FROM analysis_tasks WHERE id=77').fetchone()
    assert review == (None, None, None, 0, None)
