import numpy as np
import tensorflow as tf


def load_resnet50_places365_model(input_shape, weights, include_top, pooling):
    """
    Returns a resnet50 tf.keras.models.Model with an input_shape of (244, 224, 3)
    """
    model = tf.keras.models.load_model(
        "/home/ss3/josh/background/video_clsf/utils/places365/resnet50_places365", compile=True)
    return model


def preprocess_input(image_rgb):
    image_rgb = np.array(image_rgb) / 255.
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_whitened_np = (image_rgb - mean) / std
    return img_whitened_np
