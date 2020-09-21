import os
from fastapi.testclient import TestClient
from app.main import app

os.chdir('app/')

def test_valid_defaults():
    with TestClient(app) as client:
        url = "/classify"
        response = client.get(url)
        assert response.status_code == 200
        assert response.json()['result'] in ['0', '1']

def test_valid_input():
    with TestClient(app) as client:
        url = "/classify/?numeric0=1&categorical0=a"
        response = client.get(url)
        assert response.status_code == 200
        assert response.json()['result'] in ['0', '1']

def test_invalid_input():
    with TestClient(app) as client:
        url = "/classify/?numeric0=1&categorical0=d"
        response = client.get(url)
        assert response.status_code == 422