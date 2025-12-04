import numpy as np
from PIL import Image

from enhancekit import utils


def test_load_image_scales_float_numpy_array():
    array = np.ones((2, 2, 3), dtype=np.float32) * 0.5
    image = utils.load_image(array)
    assert isinstance(image, Image.Image)
    np.testing.assert_array_equal(np.array(image), np.full((2, 2, 3), 127, dtype=np.uint8))


def test_load_image_preserves_uint8_numpy_array():
    array = np.array([[[0, 128, 255]]], dtype=np.uint8)
    image = utils.load_image(array)
    assert isinstance(image, Image.Image)
    np.testing.assert_array_equal(np.array(image), array)


def test_load_image_handles_chw_numpy_tensor():
    array = np.ones((3, 2, 2), dtype=np.uint8) * 255
    image = utils.load_image(array)
    assert isinstance(image, Image.Image)
    np.testing.assert_array_equal(np.array(image), np.ones((2, 2, 3), dtype=np.uint8) * 255)
