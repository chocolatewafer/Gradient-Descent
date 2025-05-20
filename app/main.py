from fastapi import FastAPI
from app.core.config import APP_NAME, API_PREFIX, VERSION
from app.api.routes.api import router


def start_application() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=VERSION)
    app.include_router(router, prefix=API_PREFIX)
    return app


my_app = start_application()
