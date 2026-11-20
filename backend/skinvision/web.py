"""Same-origin browser deployment without masking missing API/static routes."""
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from starlette.routing import Match


class BrowserRoute(APIRoute):
    def matches(self, scope):
        match, child_scope = super().matches(scope)
        path = child_scope.get('path_params', {}).get('path', '')
        if path.split('/')[0] in {'api', 'static', 'health'}:
            return Match.NONE, {}
        return match, child_scope


def mount_frontend(app, directory):
    root = Path(directory).resolve()
    if not (root / 'index.html').is_file():
        return

    def browser_page(path: str):
        if path.split('/')[0] in {'api', 'static', 'assets', 'health'} and not path.startswith('assets/'):
            raise HTTPException(404)
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise HTTPException(404)
        if candidate.is_file():
            return FileResponse(candidate)
        if path.startswith('assets/') or Path(path).suffix:
            raise HTTPException(404)
        return FileResponse(root / 'index.html', headers={'Cache-Control': 'no-cache'})

    app.router.routes.append(BrowserRoute('/{path:path}', browser_page,
                                         methods=['GET'], include_in_schema=False))
