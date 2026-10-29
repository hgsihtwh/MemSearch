from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app


class _OkSession:
    async def execute(self, *_args: object) -> None:
        return None


class _BrokenSession:
    async def execute(self, *_args: object) -> None:
        raise ConnectionError("база недоступна")


def _client(session: object) -> TestClient:
    async def override() -> AsyncIterator[object]:
        yield session

    app.dependency_overrides[get_session] = override
    return TestClient(app)


def test_health_ok() -> None:
    response = _client(_OkSession()).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_database_unavailable() -> None:
    response = _client(_BrokenSession()).get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"
