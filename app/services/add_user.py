import asyncio
from app.services.db import engine, AsyncSessionLocal
from app.schemas.model import User

async def add_user(username, password, role):
    async with AsyncSessionLocal() as session:
        user = User(username=username, password=password, role=role)
        session.add(user)
        await session.commit()
        print(f"✅ User '{username}' added successfully.")

if __name__ == "__main__":
    asyncio.run(add_user("Tony", "password123", "engineering"))
