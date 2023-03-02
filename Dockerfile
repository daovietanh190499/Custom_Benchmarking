ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.10-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace/benchmark

# Copy the model files
COPY . .

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace/benchmark/pyprof2

RUN pip install .

WORKDIR /workspace/benchmark

ENV CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV WANDB_API_KEY=49d00f97c2faf751e194885af42b0d9ac4196b0f
