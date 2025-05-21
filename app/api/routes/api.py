from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.core import ny
from app.core.config import API_VERSION
from pydantic import BaseModel, Field

router = APIRouter(prefix=f"/{API_VERSION}", tags=["gradient-descent"])


class Gradient_descent_data:
    x = []  # features
    y = []  # target variables
    w_in, b_in = 0, 0  # intial values
    iterations = 10000  # number of iterations to perform gradient descent
    lr = 0.001  # learning rate

    @classmethod
    def set_data(self, f, t):
        self.x = f
        self.y = t

    @classmethod
    def get_features(self):
        return self.x, self.y


class Features(BaseModel):
    x: list = Field(default=[1, 2, 3, 4], examples=["[1,2,3,4]"])
    y: list = Field(default=[1, 2, 3, 4], examples=["[1,2,3,4]"])


class Settings(BaseModel):
    w_in: float
    b_in: float
    iterations: int = Field(default=10000)
    lr: float = Field(default=10e-4)


data = Gradient_descent_data()


@router.get("/")
def return_info():
    return "This is API for gradient descent. You can compute cost, gradient and perform gradient descent."


@router.post("/gradient/features")
def set_features(input_data: Features):
    x = input_data.x
    y = input_data.y
    if len(x) != len(y):
        return {"msg": "failure: x and y features need to be of same length"}
    data.set_data(x, y)
    return {"msg": "success", "features": data.x, "targets": data.y}


@router.get("/gradient/features")
def get_features():
    x, y = data.get_features()
    return {"x": x, "y": y}


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
