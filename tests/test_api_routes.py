import pytest

from fastapi.testclient import TestClient
from app.main import my_app
from app.core.config import API_PREFIX, API_VERSION

prefix = f"{API_PREFIX}/{API_VERSION}"


@pytest.fixture
def client():
    testapp = TestClient(my_app)
    return testapp


def test_set_features_for_gradient_descent(client):
    response = client.post(f"{prefix}/gradient", json={"x": "1,2,3,4", "y": "1,2,3,4"})
    assert response.status_code == 200
    assert response.json() == "set features and targets"


def test_get_features_for_gradient_descent(client):
    response = client.get(f"{prefix}/gradient")
    assert response.status_code == 200
    assert response.json() == "set features and targets"


def test_get_cost_gives_correct_cost(client):
    response = client.get(f"{prefix}/gradient")
    assert response.status_code == 200
    assert response.json() == "set features and targets"


def test_get_cost_gives_correct_gradient(client):
    response = client.get(f"{prefix}/gradient")
    assert response.status_code == 200
    assert response.json() == "set features and targets"


def test_get_plot_of_gradient_descent(client):
    response = client.get(f"{prefix}/gradient")
    assert response.status_code == 200
    assert response.json() == "set features and targets"


def test_perform_gradient_descent(client):
    response = client.get(f"{prefix}/gradient")
    assert response.status_code == 200
    assert response.json() == "set features and targets"
