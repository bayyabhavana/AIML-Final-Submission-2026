
import os
import cv2
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_bus_image_exists():
    image_path = os.path.join(PROJECT_DIR, "datasets", "bus.jpg")
    assert os.path.exists(image_path)

def test_road_image_exists():
    image_path = os.path.join(PROJECT_DIR, "datasets", "road.jpg")
    assert os.path.exists(image_path)

def test_bus_image_can_be_loaded():
    image_path = os.path.join(PROJECT_DIR, "datasets", "bus.jpg")
    image = cv2.imread(image_path)
    assert image is not None

def test_road_image_can_be_loaded():
    image_path = os.path.join(PROJECT_DIR, "datasets", "road.jpg")
    image = cv2.imread(image_path)
    assert image is not None
