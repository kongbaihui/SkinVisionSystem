from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.skinvision.web import mount_frontend


def test_spa_and_resource_routes(tmp_path):
    (tmp_path / 'index.html').write_text('<html>SkinVision</html>')
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets' / 'app.js').write_text('window.app = 1')
    app = FastAPI()
    mount_frontend(app, tmp_path)
    client = TestClient(app)
    assert client.get('/tasks/123').text == '<html>SkinVision</html>'
    assert client.get('/assets/app.js').status_code == 200
    for path in ['/api/v1/missing', '/static/missing', '/assets/missing.js', '/missing.js', '/%2e%2e/secret']:
        assert client.get(path).status_code == 404
    for method in ['post', 'put', 'delete', 'head']:
        assert getattr(client, method)('/api/v1/missing').status_code == 404


def test_api_only_without_build(tmp_path):
    app = FastAPI()
    mount_frontend(app, tmp_path)
    assert TestClient(app).get('/').status_code == 404
