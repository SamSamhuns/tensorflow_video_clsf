# set env ars from .env before importing any python libraries
from dotenv import load_dotenv, dotenv_values
load_dotenv(".env")

import os
import io
import copy
import random
import logging
import argparse
import os.path as osp
from datetime import datetime
from contextlib import redirect_stdout

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils.common import read_json, write_json, recursively_get_file_paths
from utils.common import init_obj
from utils.models import IMAGE_SIZE, PREPROCESS_FUNCS, BACKBONE_MODELS
from utils.model_builder import build_video_clsf_model, build_video_clsf_masked_model, build_video_clsf_movinet_model


class VideoClassifier():
    """Train Classifier on videos"""

    def __init__(self, config_path, resume_ckpt_path=None, learning_rate=None):
        config = read_json(config_path)

        # mixed precision training https://www.tensorflow.org/guide/mixed_precision, default=float32
        tf.keras.mixed_precision.set_global_policy(
            config["mixed_precision_global_policy"])
        print("mixed_precision global_policy set to:",
              tf.keras.mixed_precision.global_policy())

        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.callbacks_list = []
        self.preprocess_func = PREPROCESS_FUNCS[config["backbone"]]
        self.backbone_model = BACKBONE_MODELS[config["backbone"]]
        self.gru_units = config["gru_units"]
        self.MAX_FRAMES = config["data"]["max_frames_per_video"]
        self.img_size = IMAGE_SIZE[config["backbone"]]
        self.n_classes = config["data"]["num_classes"]
        self.data_extension = config["data"]["data_file_extension"]

        self.train_data_paths = recursively_get_file_paths(
            config["data"]["train_data_dir"], ext=self.data_extension)
        self.val_data_paths = recursively_get_file_paths(
            config["data"]["val_data_dir"], ext=self.data_extension)
        self.train_data_size = len(self.train_data_paths)
        self.validation_data_size = len(self.val_data_paths)

        # dump custom env vars from .env file to config.json
        self.config = config
        custom_env_vars = dotenv_values(".env")
        self.config["os_vars"] = custom_env_vars

        c_datetime = datetime.now().strftime(r"%Y%m%d_%H_%M_%S")
        self.models_save_path = osp.join(
            config["trainer"]["save_dir"], f"{config["name"]}_{config["backbone"]}_{c_datetime}")
        os.makedirs(self.models_save_path, exist_ok=True)

        self.logfile_path = osp.join(self.models_save_path, "train.log")
        print(f"Writing train logs to {self.logfile_path}")

        logging.basicConfig(filename=self.logfile_path,
                            filemode="a",
                            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S",
                            level=logging.INFO)
        self.logger = logging.getLogger("trainer")
        self.logger.info(
            f"Train log for model with backbone {config["backbone"]}")
        self.logger.info(
            f"Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))}")

        # if not none, training resumes from this checkpoint
        self.resume_ckpt = resume_ckpt_path
        # override learning_rate in config if it is provided as cmd line arg
        # this CLI override should be used when resuming from a ckpt
        if learning_rate:
            self.config["optimizer"]["args"]["learning_rate"] = learning_rate

        # set callbacks
        ckpt_filefmt = ("e{epoch:02d}_ta_{categorical_accuracy:.2f}_tl_{loss:.2f}"
                        "_va_{val_categorical_accuracy:.2f}_vl_{val_loss:.2f}.keras")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=osp.join(self.models_save_path, ckpt_filefmt),
            save_weights_only=False,
            monitor="val_categorical_accuracy",
            mode="auto",
            save_best_only=True)
        LROnPlateau = ReduceLROnPlateau(
            monitor="val_categorical_accuracy",
            factor=0.1, patience=2, verbose=0,
            mode="auto", min_delta=0.0001, cooldown=0,
            min_lr=0)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=3, verbose=0,
            mode="auto", baseline=None, restore_best_weights=False)
        log_file = open(self.logfile_path, mode="a", buffering=1)
        epoch_train_log_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log_file.write(
                f"epoch: {epoch}, loss: {logs["loss"]}, accuracy: {logs["categorical_accuracy"]}, "
                f"val_loss: {logs["val_loss"]}, val_accuracy: {logs["val_categorical_accuracy"]}\n"),
            on_train_end=lambda logs: log_file.close())

        def _update_initial_epoch(epoch):
            self.config["trainer"]["initial_epoch"] = epoch
            write_json(self.config, osp.join(
                self.models_save_path, "config.json"))
        update_initial_epoch_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: _update_initial_epoch(epoch))

        self.callbacks_list = [LROnPlateau, model_checkpoint_callback,
                               early_stopping_callback, epoch_train_log_callback,
                               update_initial_epoch_callback]

        # save updated config file to the checkpoint dir
        write_json(config, osp.join(self.models_save_path, "config.json"))

    def build_model(self):
        if self.resume_ckpt:
            self.logger.info(
                f"Warm start training from checkpoint {self.resume_ckpt}")
            self.model = tf.keras.models.load_model(
                self.resume_ckpt, compile=True)
            # To change anything except learning rate, recompilation is required
            self.model.optimizer.lr = self.config["optimizer"]["args"]["learning_rate"]
        else:
            self.logger.info("Cold start training")
            model_name = f"{self.config["backbone"]}_video_clsf"

            if "movinet" in self.config["backbone"]:
                self.model = build_video_clsf_movinet_model(model_name, self.backbone_model, self.MAX_FRAMES,
                                                            self.img_size, self.n_classes)
            else:
                self.model = build_video_clsf_model(model_name, self.backbone_model, self.MAX_FRAMES,
                                                    self.img_size, self.gru_units, self.n_classes)
            # self.model = build_video_clsf_masked_model(model_name, self.backbone_model, self.MAX_FRAMES,
            #                                            self.img_size, self.gru_units, self.n_classes)
            self.optimizer = init_obj(
                self.config, "optimizer", tf.keras.optimizers)
            self.loss = init_obj(self.config, "loss", tf.keras.losses)
            tf.config.optimizer.set_jit(True)
            self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=["categorical_accuracy"])

        # stream & print model summary to logger
        f = io.StringIO()
        with redirect_stdout(f):
            self.model.summary()
        model_summary = f.getvalue()
        self.logger.info(model_summary)
        print(model_summary)

    def data_generator_masked(self, data_paths, batch_size=32):
        """Generate batches with images from video."""
        img_size = IMAGE_SIZE[self.config["backbone"]]
        while True:
            random.shuffle(data_paths)
            number_samples = len(data_paths)

            for offset in range(0, number_samples, batch_size):
                frames_batch = []
                labels_batch = []
                masks_batch = []
                for fpath in data_paths[offset:offset + batch_size]:
                    data = np.load(fpath)
                    video_feats = data["arr"]
                    video_feats = video_feats[:self.MAX_FRAMES]
                    if len(video_feats) < self.MAX_FRAMES:
                        diff = self.MAX_FRAMES - len(video_feats)
                        w, h = video_feats.shape[1:3]
                        zero_frames = np.zeros([diff, w, h, 3])
                        video_feats = np.concatenate(
                            [video_feats, zero_frames])

                    masks_batch.append(np.asarray([1 if np.any(
                        frame) else 0 for frame in video_feats], dtype=bool))
                    preprocessed_frames = np.asarray(
                        [self.preprocess_func(cv2.resize(frame, (img_size, img_size)))
                         for frame in copy.deepcopy(video_feats)])
                    frames_batch.append(preprocessed_frames)
                    labels_batch.append(
                        self.config["CLASS_NAME_TO_LABEL"][fpath.split("/")[-2]])
                yield ((np.asarray(frames_batch), np.asarray(masks_batch)),
                       to_categorical(labels_batch, num_classes=self.n_classes))

    def data_generator(self, data_paths, batch_size=32):
        """Generate batches with images from video."""
        while True:
            random.shuffle(data_paths)
            number_samples = len(data_paths)

            for offset in range(0, number_samples, batch_size):
                frames_batch = []
                labels_batch = []
                for fpath in data_paths[offset:offset + batch_size]:
                    data = np.load(fpath)
                    video_feats = data["arr"]
                    video_feats = video_feats[:self.MAX_FRAMES]
                    if len(video_feats) < self.MAX_FRAMES:
                        diff = self.MAX_FRAMES - len(video_feats)
                        w, h = video_feats.shape[1:3]
                        zero_frames = np.zeros([diff, w, h, 3])
                        video_feats = np.concatenate(
                            [video_feats, zero_frames])
                    preprocessed_frames = np.asarray(
                        [self.preprocess_func(cv2.resize(frame, (self.img_size, self.img_size)))
                         for frame in copy.deepcopy(video_feats)])
                    frames_batch.append(preprocessed_frames)
                    labels_batch.append(
                        self.config["CLASS_NAME_TO_LABEL"][fpath.split("/")[-2]])
                yield np.asarray(frames_batch), to_categorical(labels_batch, num_classes=self.n_classes)

    def train(self):
        """Train model on video with image features."""
        self.model.fit(
            x=self.data_generator(data_paths=self.train_data_paths,
                                  batch_size=self.config["data"]["train_bsize"]),
            steps_per_epoch=self.train_data_size // self.config["data"]["train_bsize"],
            epochs=self.config["trainer"]["epochs"],
            verbose=self.config["trainer"]["verbose"],
            callbacks=self.callbacks_list,
            validation_data=self.data_generator(data_paths=self.val_data_paths,
                                                batch_size=self.config["data"]["val_bsize"]),
            validation_steps=self.validation_data_size // self.config["data"]["val_bsize"],
            shuffle=self.config["trainer"]["shuffle"],
            sample_weight=None,
            initial_epoch=self.config["trainer"]["initial_epoch"],
            validation_freq=self.config["trainer"]["val_freq"])


def main():
    parser = argparse.ArgumentParser(
        description="Train a video classifier to classify videos")
    parser.add_argument("--cfg", "--config_path", type=str, dest="config_path",
                        default="config/train_video_frames_reference.json",
                        help="Path to train config file (default: %(default)s)")
    parser.add_argument("-r", "--resume_ckpt", type=str,
                        help="Path to savedmodel dir to resume training. (default: %(default)s)")
    parser.add_argument("--lr", "--learning_rate", type=float, dest="learning_rate",
                        help="OPTIONAL: lr param to override that in config. (default: %(default)s)")
    args = parser.parse_args()

    model = VideoClassifier(
        args.config_path, args.resume_ckpt, args.learning_rate)
    model.build_model()
    model.train()


if __name__ == "__main__":
    main()
