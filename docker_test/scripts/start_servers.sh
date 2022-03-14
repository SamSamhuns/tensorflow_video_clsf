#!/bin/bash
#Run Triton on GRPC port 8081 inside docker. Keep it 8081 always
tritonserver --model-store models --allow-grpc=true --allow-http=false --grpc-port=8081 --allow-metrics=false --allow-gpu-metrics=false &
