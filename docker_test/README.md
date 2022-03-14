# Image Landmark Detection

Tensorflow Model for Image Landmark Detection.

## Build and Run Docker image.

```shell
$ bash scripts/build_docker.sh
$ bash scripts/run_docker.sh -h <EXPOSED_HTTP_PORT_NUM> -g <CUDA_GPU_DEVICE_NUMBER>
# Wait for model loading (60 seconds approx) and use the Fast API with Swagger UI to test.
```

## Testing the model through the dockerized API

Set up a data directory where sub-directories are classes which contain the images. i.e.

    data
        |_ class_1
                |_ img1
                |_ img2
                |_ ....
        |_ class_2
                |_ img1
                |_ img2
                |_ ....

Install python requirements

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install requests==2.26.0
$ pip install tqdm==4.62.3
```

Test the API on test data

```shell
$ python test_docker_api.py -sd SOURCE_DATA_PATH -r RESULT_JSON_FILE_PATH -url FAST_API_URL
```

Results for individual requests will be saved in RESULT_JSON_FILE_PATH. Make sure the FAST_API_URL points to the correct http url for the api.

Add more metrics inside the `calculate_metrics(...)` function in `test_docker_api.py`
