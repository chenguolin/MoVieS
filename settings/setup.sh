# Pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -U xformers==0.0.29.post1
# pip install nvidia-cublas-cu12==12.4.5.8  # https://github.com/InternLM/lmdeploy/issues/3297

# WandB
pip install wandb[media]

# Others
pip install -U gpustat
pip install -U -r settings/requirements.txt
pip install --upgrade imageio imageio[ffmpeg]
sudo apt-get install -y ffmpeg tmux

# GSplat
MAX_JOBS=128 pip3 install git+https://github.com/nerfstudio-project/gsplat.git@v1.5.0

# VGGT
cd /opt/tiger/SGM/extensions && git clone https://github.com/facebookresearch/vggt.git
cd /opt/tiger/SGM
