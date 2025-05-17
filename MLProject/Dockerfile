FROM continuumio/miniconda3

COPY conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml

SHELL ["conda", "run", "-n", "bigmart_env", "/bin/bash", "-c"]

COPY . /app
WORKDIR /app

CMD ["mlflow", "run", "."]
