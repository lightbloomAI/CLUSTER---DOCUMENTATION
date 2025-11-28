# HPC Workflow with Docker, Conda and Slurm (GPU Nodes, No System Installation)

This document describes a lightweight HPC workflow for running GPU-accelerated projects (e.g., Gaussian Splatting) on a cluster:

- Jobs are submitted via Slurm (`sbatch`).
- The actual work runs inside a Docker container.
- CUDA, Conda and all dependencies live inside the container and in the user’s home area.
- No system-wide installation or sudo is required (an admin only needs to enable Docker + GPU support on the nodes).


## 1. Goals

- Run GPU jobs on an HPC cluster via Slurm.
- Keep the host OS clean (no system-wide CUDA or Python packages).
- Use Docker as the runtime environment.
- Use Conda inside Docker with environments stored under the user’s directory.
- Reuse a single Docker image and Conda cache across projects.

## 2. Recommended Directory Structure

For a user, e.g. `engineer`:

```
/data/users/engineer/
├── projects/
│   └── <project_name>/
└── gs-conda/      # persistent conda cache and environments
```

Example for Gaussian Splatting:

```
/data/users/engineer/projects/gaussian-splatting
/data/users/engineer/gs-conda
```

## 3. Dockerfile (Base Image)

Create this file under your project directory:

```
/data/users/<user>/projects/gaussian-splatting/Dockerfile
```

```
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y     git     wget     build-essential     cmake     && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh &&     bash /tmp/miniconda.sh -b -p $CONDA_DIR &&     rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN useradd -m -s /bin/bash appuser
USER appuser
WORKDIR /workspace

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CMD ["/bin/bash"]
```

## 4. Build the Docker Image

Run:

```
cd /data/users/engineer/projects/gaussian-splatting
docker build -t gaussian-splatting-cuda .
```

## 5. Create the Conda Cache Directory

```
mkdir -p /data/users/engineer/gs-conda
```

This gives persistent Conda environments across jobs.

## 6. Optional: Interactive Debugging

```
docker run --gpus all --rm -it   -v /data/users/engineer/projects/gaussian-splatting:/workspace/gaussian-splatting   -v /data/users/engineer/gs-conda:/home/appuser/.conda   gaussian-splatting-cuda
```

Inside:

```
cd /workspace/gaussian-splatting
. /opt/conda/etc/profile.d/conda.sh
conda env create --file environment.yml   # first time only
conda activate gaussian_splatting
```

## 7. Creating the Conda Environment (first time only)

Inside the container:

```
cd /workspace/gaussian-splatting
. /opt/conda/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate gaussian_splatting
```

## 8. Daily Usage

Inside Docker (Slurm or interactive):

```
cd /workspace/gaussian-splatting
. /opt/conda/etc/profile.d/conda.sh
conda activate gaussian_splatting
```

## 9. Mount Extra Data

```
docker run --gpus all --rm -it   -v /data/users/engineer/projects/gaussian-splatting:/workspace/gaussian-splatting   -v /data/users/engineer/gs-conda:/home/appuser/.conda   -v /data/users/engineer/data:/workspace/data   gaussian-splatting-cuda
```

Inside the container:

```
/workspace/data/<dataset_name>
```

## 10. HPC Integration with Slurm (sbatch + Docker)

### Example Slurm Script

```
#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --job-name=gs-bush6
#SBATCH --output=logs/gs-bush6-%j.out
#SBATCH --error=logs/gs-bush6-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -e

cd /data/users/engineer/projects/gaussian-splatting
mkdir -p logs

docker run --gpus all --rm   -v /data/users/engineer/projects/gaussian-splatting:/workspace/gaussian-splatting   -v /data/users/engineer/gs-conda:/home/appuser/.conda   gaussian-splatting-cuda   bash -lc '
    cd /workspace/gaussian-splatting
    . /opt/conda/etc/profile.d/conda.sh
    conda activate gaussian_splatting

    python train.py       -s data/forestPlaza       -m outputs/forestPlaza       --iterations 50000       --save_iterations 1000 3000 50000       --checkpoint_iterations 1000 3000 50000       -r 2
  '
```

Submit the job:

```
sbatch run_gs_sbatch.sh
```
