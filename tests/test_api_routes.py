import pytest

from fastapi.testclient import TestClient
from app.main import my_app
from app.core.config import API_PREFIX, API_VERSION

prefix = f"{API_PREFIX}/{API_VERSION}"


@pytest.fixture
def client():
    testapp = TestClient(my_app)
    return testapp


def test_api_info_message(client):
    response = client.get(f"{prefix}/")
    assert response.status_code == 200
    assert (
        response.json()
        == "This is API for gradient descent. You can compute cost, gradient and perform gradient descent."
    )


def test_set_features_for_gradient_descent(client):
    response = client.post(
        f"{prefix}/gradient/features", json={"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]}
    )
    assert response.status_code == 200
    assert response.json() == {
        "msg": "success",
        "features": [1, 2, 3, 4],
        "targets": [1, 2, 3, 4],
    }


def test_set_features_with_diff_length_gradient_descent(client):
    response = client.post(
        f"{prefix}/gradient/features", json={"x": [1, 2, 3], "y": [1, 2, 3, 4]}
    )
    assert response.status_code == 200
    assert response.json() == {
        "msg": "failure: x and y features need to be of same length"
    }


def test_get_features_for_gradient_descent(client):
    response = client.get(f"{prefix}/gradient/features")
    assert response.status_code == 200
    assert response.json() == {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]}


def test_get_cost_gives_correct_cost(client):
    response = client.get(f"{prefix}/cost")
    assert response.status_code == 200
    assert response.json() == {"weight": 0, "bias": 0, "cost": 3.75}


def test_get_gradient_gives_correct_gradient(client):
    response = client.get(f"{prefix}/gradient")
    assert response.status_code == 200
    assert response.json() == {"dj_dw": -4.0, "dj_db": -2.5}


def test_get_plot_of_gradient_descent(client):
    client.get(f"{prefix}/gradient/descent")
    response = client.get(f"{prefix}/gradient/plot")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_perform_gradient_descent(client):
    response = client.get(f"{prefix}/gradient/descent")
    assert response.status_code == 200
    assert response.json() == {
        "msg": "params and cost after gradient descent",
        "w": 0.9937,
        "b": 0.0232,
        "cost": 5.2709e-05,
    }


def test_set_settings(client):
    response = client.post(
        f"{prefix}/gradient/settings",
        json={"w_in": 2, "b_in": 2, "lr": 0.002, "iterations": 10000},
    )
    assert response.status_code == 200
    assert response.json() == {
        "msg": "settings updated sucessfully",
        "w_in": 2,
        "b_in": 2,
        "lr": 0.002,
        "iterations": 10000,
    }


def test_perform_gradient_descent_with_new_settings(client):
    response = client.get(f"{prefix}/gradient/descent")
    assert response.status_code == 200
    assert response.json() == {
        "msg": "params and cost after gradient descent",
        "w": 0.9995,
        "b": 0.0018,
        "cost": 3.2640e-07,
    }
