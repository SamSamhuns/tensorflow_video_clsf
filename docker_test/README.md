# Dockerized API Test

## Build and Run Docker image.

```shell
$ bash scripts/build_docker.sh
$ bash scripts/run_docker.sh -h <EXPOSED_HTTP_PORT_NUM> -g <CUDA_GPU_DEVICE_NUMBER>
# Wait for model loading (60 seconds approx) and use the Fast API with Swagger UI to test.
```
