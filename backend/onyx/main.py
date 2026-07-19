import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .api import router
from .queue import worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Rendered files must be readable over SMB/NFS by other users; a restrictive
# container umask otherwise creates 0640 output that network shares can't
# serve. 0o002 -> new files are 0664 (world-readable).
os.umask(0o002)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    if config.WORKER_ENABLED:
        worker.start()
    yield
    if config.WORKER_ENABLED:
        await worker.stop()


app = FastAPI(title="Onyx", version=config.VERSION, lifespan=lifespan)
app.include_router(router)

if (config.STATIC_DIR / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=config.STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        return FileResponse(config.STATIC_DIR / "index.html")


def serve() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.PORT)


if __name__ == "__main__":
    serve()
