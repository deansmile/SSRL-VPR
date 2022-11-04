# SSRL-VPR

## 1. Set up the environment
[HPC application form link](https://www.nyu.edu/life/information-technology/research-computing-services/high-performance-computing/high-performance-computing-nyu-it/hpc-accounts-and-eligibility.html)

Using VSCode with Remote Development Extension is recommended, which enables you to conveniently edit code on the HPC server. If you want to use VSCode, then you need to connect to [NYU-VPN](https://nyu.service-now.com/sp?id=search&spa=1&q=vpnmfa) everytime before you login to HPC. After connected to NYU-VPN, open VSCode and press F1. Search for Remote-SSH: Connect to Host. Enter your_netid@greene.hpc.nyu.edu and your password to login.

Login to the Greene cluster your_netid@greene.hpc.nyu.edu

For faster installation, we want to use the computation node by running an interactivate job.
```
srun -t30:00 -c4 --mem=3000 --pty /bin/bash
```
Create a directory for the environment.
```
mkdir /scratch/$USER/environments
cd /scratch/$USER/environments
```
Copy the overlay images to the directory and unzip it (here we use the image that has 15GB spaces and can hold 500k files). Then rename the image
```
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
mv overlay-15GB-500K.ext3 habitat.ext3
```
Launch the Singularity container in read/write mode (with the :rw flag). 
```
singularity exec --overlay habitat.ext3:rw /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash
```
The above starts a bash shell inside the referenced Singularity Container overlayed with the 15GB 500K you set up earlier. This creates the functional illusion of having a writable filesystem inside the typically read-only Singularity container.

Now, inside the container, download and install miniconda to /ext3/miniconda3
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
```
Next, create a wrapper script /ext3/env.sh, and open it with vim
```
touch /ext3/env.sh
vim /ext3/env.sh
```
Now you are in the interface of vim. Press `i` to change to insert mode. Copy the following text and paste it to the interface:
```
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PATH=/ext3/miniconda3/envs/habitat/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/envs/habitat:$PATH
```
Press `esc` to exit insert mode. Type `:wq` to save and exit the file.

The wrapper script will activate your conda environment, to which you will be installing your packages and dependencies. 

Activate your conda environment with the following:
```
source /ext3/env.sh
```
Create a new conda environment for habitat:
```
conda create -n habitat python=3.7 cmake=3.14.0 -y
conda activate habitat
conda uninstall pytorch
```
Install packages:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
-f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
pip install faiss-cpu --no-cache
pip install h5py numpy scikit-learn tensorboardX scipy
pip install timm==0.3.2
conda install -c vissl apex
```
Install vissl:
```
git clone --recursive https://github.com/facebookresearch/vissl.git
cd vissl
git checkout v0.1.6
git checkout -b v0.1.6
pip install --progress-bar off -r requirements.txt
pip install opencv-python
pip uninstall -y classy_vision
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
pip uninstall -y fairscale
pip install fairscale==0.4.6
pip install -e ".[dev]"
# verify installation
python -c 'import vissl, apex'
```

Now, the environment is set up. enter `exit` twice to exit the singularity and end the interactive job.

## 2.Download
Download the code:
```
cd /scratch/$USER
git clone https://github.com/deansmile/SSRL-VPR.git
```
Download data:
```
scp deansheng@128.122.136.119:/data_1/washington-square/2.zip ./
password: Yy201023!
unzip 2.zip
```
Download pretrained model:
```
cd SSRL-VPR/mae-NetVlad
scp deansheng@128.122.136.119:/home/deansheng/mae_visualize_vit_large_ganloss.pth ./
password: Yy201023!
```

## 3. Prepare & Run experiments
1. **Do not run python files directly in the terminal. Edit the content after "python" on line 16 in test.SBATCH and submit the job.**
2. Submit job:
   ```
   cd /scratch/$USER/SSRL-VPR
   sbatch test.SBATCH
   ```
3. Check job progress
   ```
   squeue -u your_netid
   ```
4. If your job is running (R), console output of your job will be in a slurm-***.out file. Refresh your explorer to check the output.

### Preparation
Edit sub/netvlad_mat.py, mae-NetVLAD/main.py, mae-NetVLAD/pittsburgh.py, mae-NetVLAD/mae_tl.py, pytorch-NetVLAD/pittsburgh.py, pytorch-NetVLAD/main.py: Search for "ds5725" and replace all "ds5725" with your netid. 

Then run netvlad_mat.py to generate mat files for NetVLAD input.
```
python sub/netvlad_mat.py
```

### CNN-NetVLAD
Make sure you delete the folder checkpoints in pytorch-NetVlad before running new experiment.

#### Cluster
```
python pytorch-NetVlad/main.py --mode=cluster --pooling=netvlad --num_clusters=64
```
#### Train
```
python pytorch-NetVlad/main.py --mode=train --pooling=netvlad --num_clusters=64
```
#### Test
```
python pytorch-NetVlad/main.py --mode=test
```

### MAE-NetVLAD
Make sure you delete the folder checkpoints in mae-NetVlad before running new experiment.

To fix weight of the pretrained model, uncomment line 411-413 in mae-NetVlad/main.py.

#### Cluster
```
python mae-NetVlad/main.py --mode=cluster --pooling=netvlad --num_clusters=64
```
#### Train
```
python mae-NetVlad/main.py --mode=train --pooling=netvlad --num_clusters=64
```
#### Test
```
python mae-NetVlad/main.py --mode=test
```

### VISSL-NetVLAD
Make sure you delete the folder checkpoints in mae-NetVlad before running new experiment.
To change SSRL method used, modify line 27, 63-64, 406-408.

#### Cluster
```
python vissl-NetVlad/main.py --mode=cluster --pooling=netvlad --num_clusters=64
```
#### Train
```
python vissl-NetVlad/main.py --mode=train --pooling=netvlad --num_clusters=64
```
#### Test
```
python vissl-NetVlad/main.py --mode=test
```
