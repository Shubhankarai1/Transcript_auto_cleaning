import logging

from fastapi import FastAPI

from routers.catalog import router as catalog_router
from routers.chat import router as chat_router
from routers.health import router as health_router
from routers.v1_auth import router as auth_router
from routers.v1_profiles import router as profiles_router


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.include_router(health_router)
app.include_router(catalog_router)
app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(profiles_router)
