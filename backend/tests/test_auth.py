from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    resp = await client.post("/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "secret123",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient):
    await client.post("/auth/register", json={
        "username": "dupuser",
        "email": "dup1@example.com",
        "password": "secret123",
    })
    resp = await client.post("/auth/register", json={
        "username": "dupuser",
        "email": "dup2@example.com",
        "password": "secret123",
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    await client.post("/auth/register", json={
        "username": "loginuser",
        "email": "login@example.com",
        "password": "secret123",
    })
    resp = await client.post("/auth/login", json={
        "username": "loginuser",
        "password": "secret123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    resp = await client.post("/auth/login", json={
        "username": "nonexistent",
        "password": "wrong",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
