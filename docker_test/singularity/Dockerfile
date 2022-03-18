FROM tensorflow/tensorflow:latest-gpu

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install opencv & python-venv reqs
RUN apt-get update \
 && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv -y

# set work directory
WORKDIR /tensorflow_training

# setup python env vars & virtual env
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r requirements.txt
