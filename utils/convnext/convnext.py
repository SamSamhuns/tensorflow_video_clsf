import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MODEL_HANDLE_MAP = {
    "convnext_tiny_1k_224_fe": "https://tfhub.dev/sayakpaul/convnext_tiny_1k_224_fe/1",
    "convnext_small_1k_224_fe": "https://tfhub.dev/sayakpaul/convnext_small_1k_224_fe/1",
    "convnext_base_1k_224_fe": "https://tfhub.dev/sayakpaul/convnext_base_1k_224_fe/1",
    "convnext_base_1k_384_fe": "https://tfhub.dev/sayakpaul/convnext_base_1k_384_fe/1",
    "convnext_large_1k_224_fe": "https://tfhub.dev/sayakpaul/convnext_large_1k_224_fe/1",
    "convnext_large_1k_384_fe": "https://tfhub.dev/sayakpaul/convnext_large_1k_384_fe/1",
    "convnext_base_21k_1k_224_fe": "https://tfhub.dev/sayakpaul/convnext_base_21k_1k_224_fe/1",
    "convnext_base_21k_1k_384_fe": "https://tfhub.dev/sayakpaul/convnext_base_21k_1k_384_fe/1",
    "convnext_large_21k_1k_224_fe": "https://tfhub.dev/sayakpaul/convnext_large_21k_1k_224_fe/1",
    "convnext_large_21k_1k_384_fe": "https://tfhub.dev/sayakpaul/convnext_large_21k_1k_384_fe/1",
}


def load_convnext_model(backbone_model, input_shape, **kwargs):
    if backbone_model not in MODEL_HANDLE_MAP:
        raise NotImplementedError(
            f"ConvNext model with id {backbone_model} not implemented")
    hub_url = MODEL_HANDLE_MAP[backbone_model]

    encoder = hub.KerasLayer(hub_url, trainable=True)
    model = tf.keras.Sequential([encoder])
    # required for building the model
    example_input = tf.ones([1, *input_shape])
    model(example_input)
    return model


def preprocess_input(image_rgb):
    image_rgb = np.array(image_rgb) / 255.
    img_whitened_np = (image_rgb - MEAN) / STD
    return img_whitened_np
