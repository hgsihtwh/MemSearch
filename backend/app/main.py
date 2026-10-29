from typing import Annotated

from fastapi import Depends, FastAPI, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session

app = FastAPI(title="MemSearch API", version="0.1.0")


@app.get("/api/v1/health", tags=["Сервис"])
async def health(
    session: Annotated[AsyncSession, Depends(get_session)], response: Response
) -> dict[str, str]:
    """Проверка работоспособности: сервис отвечает и база данных доступна."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "error", "database": "unavailable"}
    return {"status": "ok", "database": "ok"}
