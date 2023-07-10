################################################# Image #################################################
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

############################################# Date and Time #############################################
ENV TZ="America/Sao_Paulo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update -y && apt-get install -y locales curl wget tar sudo git apt-utils \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

############################################# DEPENDENCIES ##############################################
RUN apt-get install -y gcc g++ graphviz vim python3 python3-pip python3-dev automake make \
    build-essential cmake llvm-dev clang cython3 python-is-python3 libedit-dev libxml2-dev libtinfo-dev \
    python3-setuptools

RUN pip3 install numpy==1.20.3 decorator scipy pytest psutil typing_extensions tornado cloudpickle \ 
    'xgboost<1.6.0' synr mxnet 
RUN pip3 install transformers==3.0 torch==1.13 tensorflow==2.13.0 tensorrt==8.5.1.7

########################################### COPY HAMSA ##################################################
RUN cd $HOME && git clone https://github.com/lac-dcc/garkain 

########################################### INSTALL TVM #################################################
RUN cd $HOME && git clone -b v0.10.0 --recursive https://github.com/apache/tvm && \
    cd tvm && mkdir -p build && \
    cp $HOME/garkain/docker/cuda/config.cmake $HOME/tvm/build && \
    cp $HOME/garkain/docker/droplet_algorithm/*.py $HOME/tvm/python/tvm/autotvm/tuner/ && \ 
    cd build && cmake .. && make -j4 && \
    cd .. && sudo make cython3

RUN echo "export PYTHONPATH=/root/tvm/python:/python:" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.0/compat/" >> ~/.bashrc

########################################### WORKDIR #####################################################
WORKDIR /root/garkain/src/tvm/
ENTRYPOINT ["/bin/bash"]
