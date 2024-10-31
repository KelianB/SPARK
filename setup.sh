conda create -n SPARK python=3.9
conda activate SPARK

#################### MultiFLARE ####################

conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install iopath -c iopath -y

pip install ninja
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install gpytoolbox opencv-python meshzoo trimesh matplotlib chumpy lpips configargparse open3d wandb
pip install xatlas
pip install git+https://github.com/jonbarron/robust_loss_pytorch

#################### TrackerAdaptation ####################

# Many of these dependencies are not used - they are here so we can import the code of DECA/EMOCA without errors.
pip install mediapipe==0.10.11
pip install timm~=0.9.16 adabound~=0.0.5 compress-pickle~=1.2.0 face-alignment==1.3.4 facenet-pytorch~=2.5.1 imgaug==0.4.0 albumentations==1.4.8 scikit-video==1.1.11
conda install omegaconf~=2.0.6 pytorch-lightning==1.4.9 torchmetrics==0.6.2 hickle==5.0.2 munch~=2.5.0 torchfile==0.1.0 -c conda-forge

############################################################

# We need a specific version of numpy to pickle-load FLAME.
# However, our dependencies have likely upgraded numpy, so we need to reinstall the correct version.
# This will probably throw a warning.
pip install numpy==1.23

# For downloading pre-trained models
pip install gdown
