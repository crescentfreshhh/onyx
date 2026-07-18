import asyncio
import logging
import time
from typing import Optional

from . import db, media, pipeline
from .models import JobSettings

log = logging.getLogger("onyx.queue")


class Worker:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._cancels: dict[int, asyncio.Event] = {}

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def notify(self) -> None:
        self._wake.set()

    def cancel_job(self, job_id: int) -> bool:
        event = self._cancels.get(job_id)
        if event:
            event.set()
            return True
        job = db.get_job(job_id)
        if job and job["status"] == "queued":
            db.update_job(job_id, status="canceled", finished_at=time.time())
            return True
        return False

    async def _loop(self) -> None:
        while True:
            job = db.next_queued_job()
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=10)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._run_job(job)

    async def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        cancel = asyncio.Event()
        self._cancels[job_id] = cancel
        db.update_job(job_id, status="running", started_at=time.time(), error=None)
        log.info("job %d started: %s", job_id, job["input_path"])

        last_write = 0.0

        async def on_progress(progress: float, fps: Optional[float], eta: Optional[float]) -> None:
            nonlocal last_write
            now = time.monotonic()
            if now - last_write < 1.0:
                return
            last_write = now
            fields: dict = {}
            if progress >= 0:
                fields["progress"] = round(progress, 4)
            if fps is not None:
                fields["fps"] = fps
            if eta is not None:
                fields["eta_seconds"] = round(eta)
            if fields:
                db.update_job(job_id, **fields)

        try:
            info = await media.probe(job["input_path"])
            if info is None:
                raise RuntimeError("could not probe input file")
            settings = JobSettings.model_validate(job["settings"])
            await pipeline.run(
                job["input_path"],
                job["output_path"],
                settings,
                info["duration"],
                on_progress,
                cancel,
            )
            db.update_job(job_id, status="completed", progress=1.0, eta_seconds=None,
                          finished_at=time.time())
            log.info("job %d completed", job_id)
        except asyncio.CancelledError:
            db.update_job(job_id, status="canceled", finished_at=time.time())
            log.info("job %d canceled", job_id)
        except Exception as exc:
            db.update_job(job_id, status="failed", error=str(exc), finished_at=time.time())
            log.exception("job %d failed", job_id)
        finally:
            self._cancels.pop(job_id, None)


worker = Worker()
