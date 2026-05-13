from __future__ import annotations

from dataclasses import asdict

from config import Settings


def create_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Missing dependency: fastapi. Install requirements.txt first.") from exc

    class DraftRequest(BaseModel):
        query: str
        max_rounds: int = 2
        top_k: int = 6

    app = FastAPI(title="国企公文 Agentic Workflow API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/settings")
    def settings_preview() -> dict:
        settings = Settings.load()
        data = asdict(settings)
        data["llm_api_key"] = "***" if data.get("llm_api_key") else None
        data["chroma_dir"] = str(data["chroma_dir"])
        data["bm25_dir"] = str(data["bm25_dir"])
        return data

    @app.post("/draft")
    async def draft(_: DraftRequest) -> dict:
        return {
            "status": "not_implemented",
            "message": "CLI is the primary delivery path in this phase. Wire this endpoint to AgenticWorkflow next.",
        }

    return app


app = create_app()
