# Use use previous versions, modify these variables
# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"

ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas




# Install Miniconda (already installed)
# RUN apt-get update
# RUN apt-get install -y curl
# ENV CONDA_DIR=/opt/conda
# RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
#     && bash miniconda.sh -b -p $CONDA_DIR \
#     && rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment.yaml and create conda environment
WORKDIR /workspace
COPY environment.yaml .

# Create env and pin its name
RUN conda env create -f environment.yaml \
    && conda clean -afy

# Extract environment name from environment.yaml
# (this will ensure the container always activates it)
ARG ENV_NAME
RUN export ENV_NAME=$(head -n 1 environment.yaml | cut -d' ' -f2) \
    && echo "conda activate $ENV_NAME" >> ~/.bashrc

# Use bash with conda environment as default shell
SHELL ["/bin/bash", "--login", "-c"]

# Entrypoint
CMD ["/bin/bash"]