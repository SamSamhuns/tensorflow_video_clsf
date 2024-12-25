#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 -h http_port"
   echo -e "\t-h http"
   exit 1 # Exit script after printing help
}

while getopts "h:" opt
do
   case "$opt" in
      h ) http="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$http" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

echo "Running docker with exposed fastapi port: $http"
docker run --rm -ti \
      --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
      --name docker_api_test_container \
      -p $http:8080 \
      docker_api_test \
      bash
