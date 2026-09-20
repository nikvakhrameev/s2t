"""HTTP API. Keeps the models warm; accepts any audio format ffmpeg can read."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .audio import AudioDecodeError
from .pipeline import Engine


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="s2t", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        cleanup: bool | None = Form(None),
    ) -> dict:
        data = await file.read()
        try:
            result = await asyncio.wrap_future(
                engine.submit(data, language, cleanup, origin="api", file=file.filename)
            )
        except AudioDecodeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return result.to_dict()

    return app
