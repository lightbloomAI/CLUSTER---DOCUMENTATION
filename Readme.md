![Ishimura first node](IMG_4931 2.png)


# HPC Workflow with Docker, Conda and Slurm (GPU Nodes, No System Installation)

This document describes a lightweight HPC workflow for running GPU‑accelerated projects (e.g., Gaussian Splatting) on a cluster.

The design goal is:

- Users do **not** install anything system‑wide.
- All CUDA, Conda, Python environments, and datasets live **inside Docker containers** and under each user’s `/data/users/<user>/` directory.
- Workloads are executed via **Slurm (`sbatch`)** on GPU nodes.
- Everything remains clean, reproducible, and scalable.

---

## 0. Connecting to the Server

You can connect **from terminal** or **using Visual Studio Code Remote SSH**.

### Terminal access

```
ssh <user>@<server_ip>
```

Enter your password.

If you don’t know your credentials, ask **Kevin**.

---

### VSCode Remote SSH (Optional)

Add this to your local SSH config:

```
Host ishimura
    HostName askForIP
    User YourUser
    IdentityFile ~/.ssh/id_rsa
```

Then connect:

```
ssh ishimura
```

VSCode will request your SSH password if needed.

---

## 1. Workspace Structure

Once inside the server, navigate to your personal workspace:

```
cd /data/users/<your_user>/
```

Work **inside the `projects/` directory**:

```
cd projects
```

Clone all repositories you need:

```
git clone https://github.com/your-org/gaussian-splatting.git
git clone https://github.com/your-org/other-project.git
```

---

## 2. Recommended Directory Structure

```
/data/users/<user>/
├── projects/
│   └── gaussian-splatting/
└── gs-conda/          # persistent conda environments and package cache
```

---

## 3. Dockerfile (Base Image)

Create:

```
/data/users/<user>/projects/gaussian-splatting/Dockerfile
```

```
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git wget build-essential cmake && rm -rf /var/lib/apt/lists/*

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

---

## 4. Build the Docker Image

```
cd /data/users/<user>/projects/gaussian-splatting
docker build -t gaussian-splatting-cuda .
```

---

## 5. Conda Cache Directory

```
mkdir -p /data/users/<user>/gs-conda
```

---

## 6. Optional Interactive Run

```
docker run --gpus all --rm -it   -v /data/users/<user>/projects/gaussian-splatting:/workspace/gaussian-splatting   -v /data/users/<user>/gs-conda:/home/appuser/.conda   gaussian-splatting-cuda
```

Inside:

```
cd /workspace/gaussian-splatting
. /opt/conda/etc/profile.d/conda.sh
conda env create --file environment.yml
conda activate gaussian_splatting
```

---

## 7. Daily Workflow

```
cd /workspace/gaussian-splatting
. /opt/conda/etc/profile.d/conda.sh
conda activate gaussian_splatting
```

---

## 8. Mounting Additional Data

```
docker run --gpus all --rm -it   -v /data/users/<user>/projects/gaussian-splatting:/workspace/gaussian-splatting   -v /data/users/<user>/gs-conda:/home/appuser/.conda   -v /data/users/<user>/data:/workspace/data   gaussian-splatting-cuda
```

Inside:

```
/workspace/data/<dataset_name>
```

---

## 9. HPC Integration with Slurm (sbatch + Docker)

### Example Slurm Script

Create `run_gs_sbatch.sh`:

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

Submit:

```
sbatch run_gs_sbatch.sh
```

---

## 10. Optional: Dataset Extraction via AWS S3

Example extractor:

```
import boto3

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = ""

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def download():
    bucket_name = "cenarius"
    object_key = "gaussian/gaussian_data.zip"
    local_path = "/data/users/engineer/projects/gaussian-splatting/data/gaussian_data.zip"

    print("Starting download...")
    s3.download_file(bucket_name, object_key, local_path)
    print(f"Download finished: {local_path}")

if __name__ == "__main__":
    print("Gaussian data downloader")
    print("------------------------")
    download()
    print("You can now run the training script using that data.")
```

If you need AWS configuration, ask **Zoltan**.

