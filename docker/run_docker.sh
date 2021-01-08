#!/bin/bash

cd /home/file_server/nishimura/unet_pytorch

docker run --gpus=all --rm -it -p 8888:8888 --name root -v $(pwd):/workdir -e PASSWORD=password -w workdir naivete5656/pytorch
