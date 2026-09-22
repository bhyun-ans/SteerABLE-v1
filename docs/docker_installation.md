### Run with Docker

1. Install Docker (with GPU Support)

    Ensure that Docker is installed and configured with GPU support. Follow these steps:
    *   Install [Docker](https://www.docker.com/) if not already installed.
    *   Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support.
    *   Verify the setup with (using a version close to our environment):
        ```bash
        docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
        ```

2. Pull the Docker image
    This is upstream Protenix's dependency image and works unchanged for SteerABLE-v1: it contains PyTorch, HMMER, Kalign, CUTLASS and the rest, but no source code. You mount your own checkout into it.
    ```bash
    docker pull ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4
    ```

3. Clone this repository
    ```bash
    git clone https://github.com/bhyun-ans/SteerABLE-v1.git
    cd ./SteerABLE-v1
    ```

4. Run Docker with an interactive shell
    Mount the current directory to `/app` inside the container. If you have external data or weights (e.g., in `/root/protenix`), consider mounting them as well.
    ```bash
    docker run --gpus all -it \
        -v "$(pwd)":/app \
        -v /dev/shm:/dev/shm \
        ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4 \
        /bin/bash
    ```

5. Install SteerABLE-v1 and verify
    Once inside the container, install in editable mode and verify:
    ```bash
    cd /app
    pip install -e .
    
    # Verify the installation by checking the help message
    steerable-v1 --help
    ```

After completing these steps, you can proceed with inference or training. See [Inference Guide](infer_json_format.md) for more details.
