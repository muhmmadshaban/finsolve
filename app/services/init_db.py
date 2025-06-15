# app/services/init_db.py

import asyncio
from app.schemas.model import Base
from app.services.db import engine

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(create_tables())
