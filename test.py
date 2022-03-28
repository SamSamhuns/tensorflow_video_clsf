# set env ars from .env before importing any python libraries
from dotenv import load_dotenv
load_dotenv(".env")

import os
import io
import logging
import argparse
import os.path as osp
from functools import partial
from datetime import datetime
from collections import OrderedDict
from contextlib import redirect_stdout

import cv2
import tqdm
import numpy as np
import tensorflow as tf

from utils.common import read_json, write_json, recursively_get_file_paths
from utils.model import PREPROCESS_FUNCS, IMAGE_SIZE
from utils.metrics import accuracy, precision, recall, f1score, acc_per_class
from utils.metrics import confusion_matrix, top_k_acc, plot_confusion_matrix, classification_report_sklearn


def test(config, model_path):
    exp_name = config["name"]
    backbone = config["backbone"]
    test_data_path = config["data"]["test_data_dir"]
    num_classes = config["data"]["num_classes"]
    MAX_FRAMES = config["data"]["max_frames_per_video"]
    LABEL_MAP = OrderedDict(config["CLASS_NAME_TO_LABEL"])

    # NOTE: If there are multiple classes with same label,
    # the class with the first repeated label is used
    # and the order of class_names: class_label will matter in LABEL_MAP
    _filtered_label_map, _seen_val_set = {}, set()
    for key, val in LABEL_MAP.items():
        if val not in _seen_val_set:
            _filtered_label_map[key] = val
            _seen_val_set.add(val)
    LABEL_MAP = _filtered_label_map

    c_datetime = datetime.now().strftime(r'%Y%m%d_%H_%M_%S')
    log_dir = osp.join(config["trainer"]["log_dir"],
                       f"{exp_name}_{backbone}_{c_datetime}")
    logfile_path = osp.join(log_dir, "test.log")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Writing test logs to {logfile_path} for model {model_path}")
    logging.basicConfig(filename=logfile_path,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("tester")
    logger.info(f"Test results for model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model._name = f"{backbone}_video_clsf"

    # stream & print model summary to logger
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary = f.getvalue()
    logger.info(model_summary)
    print(model_summary)

    data_paths = recursively_get_file_paths(
        test_data_path, ext=config["data"]["data_file_extension"])
    number_samples = len(data_paths)
    batch_size = config["data"]["test_bsize"]
    preprocess_func = PREPROCESS_FUNCS[backbone]
    img_size = IMAGE_SIZE[backbone]
    labels_seen = set()

    agg_pred, agg_label = [], []
    with tqdm.tqdm(total=number_samples) as pbar:
        for offset in range(0, number_samples, batch_size):
            frames_batch = []
            labels_batch = []
            for fpath in data_paths[offset:offset + batch_size]:
                data = np.load(fpath)
                video_feats = data['arr']
                video_feats = video_feats[:MAX_FRAMES]
                if len(video_feats) < MAX_FRAMES:
                    diff = MAX_FRAMES - len(video_feats)
                    w, h = video_feats.shape[1:3]
                    zero_frames = np.zeros([diff, w, h, 3])
                    video_feats = np.concatenate([video_feats, zero_frames])
                preprocessed_frames = np.asarray(
                    [preprocess_func(cv2.resize(frame, (img_size, img_size))) for frame in video_feats])

                frames_batch.append(preprocessed_frames)
                labels_batch.append(LABEL_MAP[fpath.split('/')[-2]])
                pbar.update(1)
            labels_seen |= set(labels_batch)
            frames_batch = np.asarray(frames_batch)
            labels_batch = np.asarray(labels_batch)
            output_batch = model(frames_batch)
            agg_pred.append(output_batch.numpy())
            agg_label.append(labels_batch)
    # shape=(len dataloader, bsize, n_cls)
    agg_pred = np.concatenate(agg_pred, axis=0)
    # shape=(len dataloader, bsize)
    agg_label = np.concatenate(agg_label, axis=0)
    # combine dataloader len & bsize axes
    agg_pred = agg_pred.reshape(-1, agg_pred.shape[-1])
    agg_label = agg_label.flatten()

    # filter label names that were not seen in test set
    LABEL_MAP = {cname: clabel for cname, clabel in LABEL_MAP.items()
                 if clabel in labels_seen}

    met_val_dict = {}
    met_func_dict = {"accuracy": accuracy,
                     "accuracy_top_2": partial(top_k_acc, k=2),
                     "accuracy_top_3": partial(top_k_acc, k=3),
                     "accuracy_top_4": partial(top_k_acc, k=4),
                     "accuracy_top_5": partial(top_k_acc, k=5),
                     "acc_per_class": partial(acc_per_class, num_classes=num_classes),
                     "precision": precision,
                     "recall": recall,
                     "f1score": f1score,
                     "confusion_matrix": partial(confusion_matrix, num_classes=num_classes),
                     "save_confusion_matrix": partial(plot_confusion_matrix,
                                                      target_names=list(
                                                          LABEL_MAP.keys()),
                                                      savepath=osp.join(log_dir, "cm.jpg")),
                     "classification_report": partial(classification_report_sklearn,
                                                      target_names=list(LABEL_MAP.keys()))}
    for met, met_func in met_func_dict.items():
        met_val_dict[met] = met_func(agg_label, agg_pred)

    logger.info(f"Classes: {LABEL_MAP}")
    log = {met: met_val for met, met_val in met_val_dict.items()}
    logger.info(f"test: {(log)}")
    print(confusion_matrix(agg_label, agg_pred, num_classes=num_classes))
    print(log)

    # save updated config file to the checkpoint dir
    write_json(config, osp.join(log_dir, 'config.json'))


def test_masked(config, model_path):
    exp_name = config["name"]
    backbone = config["backbone"]
    test_data_path = config["data"]["test_data_dir"]
    num_classes = config["data"]["num_classes"]
    MAX_FRAMES = config["data"]["max_frames_per_video"]
    LABEL_MAP = OrderedDict(config["CLASS_NAME_TO_LABEL"])

    # NOTE: If there are multiple classes with same label,
    # the class with the first repeated label is used
    # and the order of class_names: class_label will matter in LABEL_MAP
    _filtered_label_map, _seen_val_set = {}, set()
    for key, val in LABEL_MAP.items():
        if val not in _seen_val_set:
            _filtered_label_map[key] = val
            _seen_val_set.add(val)
    LABEL_MAP = _filtered_label_map

    c_datetime = datetime.now().strftime(r'%Y%m%d_%H_%M_%S')
    log_dir = osp.join(config["trainer"]["log_dir"],
                       f"{exp_name}_{backbone}_{c_datetime}")
    logfile_path = osp.join(log_dir, "test.log")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Writing test logs to {logfile_path} for model {model_path}")
    logging.basicConfig(filename=logfile_path,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("tester")
    logger.info(f"Test results for model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model._name = f"{backbone}_video_clsf"

    # stream & print model summary to logger
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary = f.getvalue()
    logger.info(model_summary)
    print(model_summary)

    data_paths = recursively_get_file_paths(
        test_data_path, ext=config["data"]["data_file_extension"])
    number_samples = len(data_paths)
    batch_size = config["data"]["test_bsize"]
    preprocess_func = PREPROCESS_FUNCS[backbone]
    img_size = IMAGE_SIZE[backbone]
    labels_seen = set()

    agg_pred, agg_label = [], []
    with tqdm.tqdm(total=number_samples) as pbar:
        for offset in range(0, number_samples, batch_size):
            frames_batch = []
            labels_batch = []
            masks_batch = []
            for fpath in data_paths[offset:offset + batch_size]:
                data = np.load(fpath)
                video_feats = data['arr']
                video_feats = video_feats[:MAX_FRAMES]
                if len(video_feats) < MAX_FRAMES:
                    diff = MAX_FRAMES - len(video_feats)
                    w, h = video_feats.shape[1:3]
                    zero_frames = np.zeros([diff, w, h, 3])
                    video_feats = np.concatenate([video_feats, zero_frames])

                masks_batch.append(np.asarray([1 if np.any(
                    frame) else 0 for frame in video_feats], dtype=bool))
                preprocessed_frames = np.asarray(
                    [preprocess_func(cv2.resize(frame, (img_size, img_size))) for frame in video_feats])
                frames_batch.append(preprocessed_frames)
                labels_batch.append(LABEL_MAP[fpath.split('/')[-2]])
                pbar.update(1)
            labels_seen |= set(labels_batch)
            frames_batch = np.asarray(frames_batch)
            labels_batch = np.asarray(labels_batch)
            masks_batch = np.asarray(masks_batch)
            output_batch = model((frames_batch, masks_batch))
            agg_pred.append(output_batch.numpy())
            agg_label.append(labels_batch)
    # shape=(len dataloader, bsize, n_cls)
    agg_pred = np.concatenate(agg_pred, axis=0)
    # shape=(len dataloader, bsize)
    agg_label = np.concatenate(agg_label, axis=0)
    # combine dataloader len & bsize axes
    agg_pred = agg_pred.reshape(-1, agg_pred.shape[-1])
    agg_label = agg_label.flatten()

    # filter label names that were not seen in test set
    LABEL_MAP = {cname: clabel for cname, clabel in LABEL_MAP.items()
                 if clabel in labels_seen}

    met_val_dict = {}
    met_func_dict = {"accuracy": accuracy,
                     "accuracy_top_2": partial(top_k_acc, k=2),
                     "accuracy_top_3": partial(top_k_acc, k=3),
                     "accuracy_top_4": partial(top_k_acc, k=4),
                     "accuracy_top_5": partial(top_k_acc, k=5),
                     "acc_per_class": partial(acc_per_class, num_classes=num_classes),
                     "precision": precision,
                     "recall": recall,
                     "f1score": f1score,
                     "confusion_matrix": partial(confusion_matrix, num_classes=num_classes),
                     "save_confusion_matrix": partial(plot_confusion_matrix,
                                                      target_names=list(
                                                          LABEL_MAP.keys()),
                                                      savepath=osp.join(log_dir, "cm.jpg")),
                     "classification_report": partial(classification_report_sklearn,
                                                      target_names=list(LABEL_MAP.keys()))}
    for met, met_func in met_func_dict.items():
        met_val_dict[met] = met_func(agg_label, agg_pred)

    logger.info(f"Classes: {LABEL_MAP}")
    log = {met: met_val for met, met_val in met_val_dict.items()}
    logger.info(f"test: {(log)}")
    print(confusion_matrix(agg_label, agg_pred, num_classes=num_classes))
    print(log)

    # save updated config file to the checkpoint dir
    write_json(config, osp.join(log_dir, 'config.json'))


def main():
    parser = argparse.ArgumentParser(description='Tensorflow Testing')
    parser.add_argument('-cp', '--config_path', type=str,
                        default="config/train_video_frames.json",
                        help="Path to train config file (default: %(default)s)")
    parser.add_argument('-mp', '--model_path', required=True, type=str,
                        help="Path to model (h5 model path or savedmodel dir)")
    args = parser.parse_args()
    config = read_json(args.config_path)
    test(config, args.model_path)
    # test_masked(config, args.model_path)


if __name__ == "__main__":
    main()
