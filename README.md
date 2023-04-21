# Video Classification Training with whole frames

## Setup

Install tensorflow from the [official docs](https://www.tensorflow.org/install/pip)

Create a `.env` file in the project home directory with the following contents with corrected paths:

    XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
    TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
    TF_CPP_MIN_LOG_LEVEL='3'
    TF_FORCE_GPU_ALLOW_GROWTH="true"
    OMP_NUM_THREADS="15"
    KMP_BLOCKTIME="0"
    KMP_SETTINGS="1"
    KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
    CUDA_DEVICE_ORDER="PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES="0"

To use virtual environment: (Must set up cuda libs and cudatoolkit manually if not set up already)

Note: When using a python virtualenv, the LD_LIBRARY_PATH variable should be set to /usr/local/cuda/lib64 in the shell source files. The XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda must also be set to the cuda directory.

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

To use conda environment: (Sets the cuda libs and cudatoolkit automatically, but cuda drivers must be set up manually if not already)

```shell
$ conda create --name tf_gpu tensorflow-gpu python=3.8 -y
$ conda activate tf_gpu
$ while read requirement; do conda install --yes $requirement; done < requirements.txt
```

## Extract Video frames and save to npz files from videos

```shell
# check for help
$ python extract_frames_from_video_dataset.py -h
# general cmd fmt
$ python -sd DATA_SRC_DIR -td DATA_SINK_DIR -mf MAX_FRAMES_TO_EXTRACT -rs RESHAPE_SIZE
# extract frames from videos in data/videos where max frames to extract is 20 and reshape size is 360,360
# -mt uses multiprocessing with 20 processes (num processes can be changed inside file)
$ python -sd data/videos -tf data/extracted_frames -mf 20 -rs 360 360 -mt
```

Train, val and test directories must contain files in npz format that contain an `arr` element storing the video frames which is of shape `VIDEO_FRAMS, HEIGHT, WIDTH, CHANNEL`.

## Train

1.  Set up appropriate tensorflow env vars in `.env` such as log level, cuda lib path, and visible devices
2.  Choose appropriate model training and hyper-parameter settings in one of the `config/*.json` files
3.  Run train cmd with `$ python train.py -cfg config/CFG_FILE.json`

Saved checkpoints will be present in `['trainer']['save_dir']` directory (By default `checkpoints`) along with the `config.json` file that was used for training.

## Test

```shell
$ python test.py -cfg PATH_TO_CONFIG -mp SAVEDMODEL_CKPT_DIR_PATH
```

Test results will be saved in `['trainer']['log_dir']` directory (By default `logs`). (Can be changed through the json config file)
