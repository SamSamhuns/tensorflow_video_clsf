from dotenv import load_dotenv
load_dotenv(".env")

import os
import argparse

import tensorflow as tf
from utils import IMAGE_SIZE, PREPROCESS_FUNCS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a video classifier to classify bg videos')
    parser.add_argument('-ml', '--model_load_path', type=str, required=True,
                        help="model load path (default: %(default)s)")
    parser.add_argument('-me', '--model_export_dir', type=str,
                        default="models",
                        help="train npz data root path (default: %(default)s)")
    parser.add_argument('-b', '--backbone', type=str,
                        default="mobilenet_v2",
                        choices=["inception_v3", "mobilenet_v2", "xception",
                                 "mobilenet_v3_large", "efficientnet_v2s", "efficientnet_v2b3"],
                        help="backbone (default: %(default)s)")
    args = parser.parse_args()
    return args


def export_model(model_load_path, model_export_dir, backbone):
    model = tf.keras.models.load_model(model_load_path, compile=False)

    model_export_dir = os.path.join(model_export_dir, backbone, "1/model.savedmodel")
    os.makedirs(model_export_dir, exist_ok=True)
    img_size = IMAGE_SIZE[backbone]
    preprocess_func = PREPROCESS_FUNCS[backbone]

    class CustomModel(tf.Module):
        def __init__(self, model, **kwargs):
            super(CustomModel, self).__init__(**kwargs)
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 15, img_size, img_size, 3), dtype=tf.uint8)])
        def update_signature(self, inp_images):  # inp_images is the input name
            # x = tf.image.central_crop(inp_images, 0.8)
            x = tf.cast(inp_images, tf.float32)
            x = preprocess_func(x)
            output = self.model(x)
            return {"predictions": output}

    custom_model = CustomModel(model)
    tf.saved_model.save(custom_model, model_export_dir,
                        signatures={"serving_default": custom_model.update_signature})


def main():
    args = parse_args()
    export_model(args.model_load_path, args.model_export_dir, args.backbone)


if __name__ == "__main__":
    main()
