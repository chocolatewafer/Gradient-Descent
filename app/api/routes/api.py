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
    cost_hist = []
    param_hist = []

    @classmethod
    def set_data(self, f, t):
        self.x = f
        self.y = t

    @classmethod
    def get_features(self):
        return self.x, self.y

    @classmethod
    def get_settings(self):
        return self.w_in, self.b_in, self.lr, self.iterations

    @classmethod
    def get_compute_data(self):
        x, y = ny.numpy_array(self.x, self.y)
        return x, y, self.w_in, self.b_in, self.lr, self.iterations

    @classmethod
    def set_hist(self, h, p):
        self.cost_hist = h
        self.param_hist = p


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
def compute_cost():
    x, y, w, b, *_ = data.get_compute_data()
    cost = float(ny.compute_cost(x, y, w, b))
    return {"weight": w, "bias": b, "cost": cost}


@router.get("/gradient")
def compute_gradient():
    x, y, w, b, *_ = data.get_compute_data()
    dw, db = ny.compute_gradient(x, y, w, b)
    return {"dj_dw": dw, "dj_db": db}


@router.get("/gradient/descent")
def gradient_descent():
    x, y, w_in, b_in, alpha, num_iters = data.get_compute_data()
    w, b, h, p = ny.gradient_descent(x, y, w_in, b_in, alpha, num_iters)
    data.set_hist(h, p)
    return {
        "msg": "params and cost after gradient descent",
        "w": w,
        "b": b,
        "cost": h[-1],
    }


@router.get("/gradient/plot")
def get_plot() -> FileResponse:
    pass
