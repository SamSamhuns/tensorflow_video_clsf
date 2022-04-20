import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def load_movinet_model(model_id='a0', num_frames=None, img_size=None):
    if model_id not in {'a0', 'a1', 'a2', 'a3', 'a4', 'a5'}:
        raise NotImplementedError(
            f"Movinet model with id {model_id} not implemented")
    hub_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/base/kinetics-600/classification/3"

    encoder = hub.KerasLayer(hub_url, trainable=True)
    inputs = tf.keras.layers.Input(shape=[num_frames, img_size, img_size, 3], dtype=tf.float32, name='image')
    outputs = encoder(dict(image=inputs))  # [batch_size, 600]
    model = tf.keras.Model(inputs, outputs)
    return model


def preprocess_input(image_rgb):
    image_rgb = np.asarray(image_rgb, dtype=np.float32) / 255.
    return image_rgb
