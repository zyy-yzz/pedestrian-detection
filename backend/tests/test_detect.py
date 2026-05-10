from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_detect_image_missing_file(client: AsyncClient):
    resp = await client.post("/detect/image")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_job_status_not_found(client: AsyncClient):
    await client.post("/auth/register", json={
        "username": "detuser",
        "email": "det@example.com",
        "password": "secret123",
    })
    login_resp = await client.post("/auth/login", json={
        "username": "detuser",
        "password": "secret123",
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.get("/detect/status/nonexistent-job-id", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_results_not_found(client: AsyncClient):
    await client.post("/auth/register", json={
        "username": "resuser",
        "email": "res@example.com",
        "password": "secret123",
    })
    login_resp = await client.post("/auth/login", json={
        "username": "resuser",
        "password": "secret123",
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.get("/results/nonexistent-job-id", headers=headers)
    assert resp.status_code == 404
