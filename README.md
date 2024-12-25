# CTRL

## Hardware Configuration

Ubuntu 16.04.3 LTS operating system with 2 * Intel Xeon E5-2660 v3 @ 2.60GHz CPU (40 CPU cores, 10 cores per physical CPU, 2 threads per core), 256 GB of RAM, and 4 * GeForce GTX TITAN X GPU with 12GB of VRAM.

## Software Configuration

```shell
conda create -n ctrl python=3.8
conda activate ctrl
pip install torch==2.1.2 numpy==1.24.1 pandas==2.0.3 scikit-learn==1.3.2 scipy==1.10.1
```

## Data Avaibility:

[IHDP](http://www.fredjo.com/)
[JOBS](http://www.fredjo.com/)


