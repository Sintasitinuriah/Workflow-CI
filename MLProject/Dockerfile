# Gunakan base image Python
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Salin project
COPY . .

# Install conda (Miniconda via Miniforge atau Mamba bisa juga digunakan)
RUN apt-get update && \
    apt-get install -y wget git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

ENV PATH="/opt/conda/bin:$PATH"

# Create conda env & install dependencies
RUN conda env create -f conda.yaml

# Aktifkan conda environment & set entrypoint
SHELL ["conda", "run", "-n", "big-mart-env", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "big-mart-env", "mlflow", "run", ".", "-P", "data_path=cleaned_data.csv"]
