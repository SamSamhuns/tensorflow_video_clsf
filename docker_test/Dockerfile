FROM nvcr.io/nvidia/tritonserver:20.12-py3

WORKDIR /cv_root

# copy Required files
COPY models models

COPY scripts/start_servers.sh .
RUN ["chmod", "+x", "start_servers.sh"]

# Start the FastAPI and Triton Servers
CMD ["/bin/bash", "start_servers.sh"]
