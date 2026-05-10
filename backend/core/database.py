from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

settings = get_settings()

# Convert sync MySQL URL to async driver robustly
_sync_url = settings.database_url
if _sync_url.startswith("mysql+pymysql://"):
    _async_url = "mysql+aiomysql://" + _sync_url[len("mysql+pymysql://"):]
elif _sync_url.startswith("mysql+mysqldb://"):
    _async_url = "mysql+aiomysql://" + _sync_url[len("mysql+mysqldb://"):]
elif _sync_url.startswith("mysql+aiomysql://"):
    _async_url = _sync_url
else:
    _async_url = _sync_url

engine = create_async_engine(_async_url, echo=settings.debug, pool_size=10, max_overflow=20)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)