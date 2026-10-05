"""Run the API (the Docker image's command): `python -m src.api.serve`.

uvicorn, stopped by a signal, raises that signal again once it has shut
down, so the process ends with code 143 for SIGTERM. Docker counts that as a
failure and `restart: on-failure` would start the API again right after a
shutdown from the web UI. Here the API stops by setting
uvicorn's own exit flag instead: no signal, exit code 0, and the container
stays stopped.

Dev mode (`just serve`, uvicorn --reload) does not go through here; there the
API stops with the other dev services when the worker exits.
"""

import os
import sys
from typing import Optional

import uvicorn

_server: Optional[uvicorn.Server] = None


def request_exit() -> bool:
    """Ask the running server to stop. False when not started through main()."""
    if _server is None:
        return False
    _server.should_exit = True
    return True


def main() -> int:
    import importlib

    config = uvicorn.Config(
        "src.api.app:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
    )
    server = uvicorn.Server(config)
    # Run as `python -m src.api.serve` this file is __main__, while the API
    # imports it as src.api.serve: a second module object. request_exit()
    # must find the server in the one the API imports.
    importlib.import_module("src.api.serve")._server = server
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
