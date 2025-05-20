from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.core import ny

router = APIRouter(prefix="/v1", tags=["gradient-descent"])


@router.post("/gradient")
def set_features():
    pass


@router.post("/settings")
def set_settings():
    pass


@router.get("/cost")
def compute_cost() -> float:
    pass


@router.get("/gradient")
def compute_gradient():
    pass


@router.get("/gradient/descent")
def gradient_descent():
    pass


@router.get("/gradient/plot")
def get_plot() -> FileResponse:
    pass
