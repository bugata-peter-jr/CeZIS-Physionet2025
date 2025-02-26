FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER bugatap@vsl.sk

## DO NOT EDIT the 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN conda install -c pytorch torchvision --yes
RUN conda install -c conda-forge pytorch-model-summary --yes
RUN conda install -c conda-forge pandas --yes
## RUN conda install -c conda-forge wfdb=4.1.2 --yes
RUN conda install -c conda-forge mne-base --yes
RUN pip install wfdb==4.1.2

