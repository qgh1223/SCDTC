FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
LABEL maintainer="nishimura"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libgtk2.0-dev \
	libgl1-mesa-glx \
	git \
	curl \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        opencv-python \
	#opencv-contrib-python \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        keras_applications \
        keras_preprocessing \
        matplotlib \
        pandas \
        scipy \
        sklearn \
	torchvision \
	pulp \
	opencv-python \
	scikit-image \
	tqdm \
    	visdom \
        && \
	python -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# TensorBoard
EXPOSE 8097
# IPython
EXPOSE 8888

WORKDIR "/notebooks"
CMD ["/run_jupyter.sh", "--allow-root"]
