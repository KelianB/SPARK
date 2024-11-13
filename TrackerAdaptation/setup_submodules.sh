#!/bin/bash

# Ensure working dir is at the location of this script
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null

echo "In order to run EMOCA, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use EMOCA, please register at:" 
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Pulling submodules"
for repo in SMIRK EMOCA
do
    echo $repo
    # Pull non-recursively & patch
    git submodule update --init submodules/$repo &&
    (cd submodules/$repo && patch -p1 <../$repo.patch)
done

# Setup EMOCA
(
    cd submodules/EMOCA
    # git submodule update --init external/SwinTransformer
    git submodule update --init --recursive external/spectre

    mkdir -p assets &&
    cd assets

    (    
        mkdir -p EMOCA/models &&
        cd EMOCA/models &&
        # EMOCA v2 model
        echo -e "\nDownloading EMOCA v2..." &&
        wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip -O EMOCA_v2_lr_mse_20.zip &&
        unzip EMOCA_v2_lr_mse_20.zip &&
        # DECA model
        echo -e "\nDownloading DECA..." &&
        wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip -O DECA.zip &&
        unzip DECA.zip &&
        rm DECA.zip EMOCA_v2_lr_mse_20.zip
    )    
    (
        mkdir -p EmotionRecognition/image_based_networks &&
        cd EmotionRecognition/image_based_networks &&
        # EmoNet
        echo -e "\nDownloading EmoNet..." &&
        wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip -O ResNet50.zip &&
        unzip ResNet50.zip &&
        rm ResNet50.zip
    )

    echo -e "\nDownloading pretrained FaceRecognition (VGG)..." &&
    wget https://download.is.tue.mpg.de/emoca/assets/FaceRecognition.zip -O FaceRecognition.zip &&
    unzip FaceRecognition.zip &&
    rm FaceRecognition.zip

    # Download FLAME for EMOCA (we can't reuse MultiFLARE's here because EMOCA needs more)
    echo -e "\nDownloading FLAME assets" &&
    wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip &&
    unzip FLAME.zip &&
    rm FLAME.zip
    
    cd ../

    (
        # Spectre / Lip reading loss
        echo -e "\nDownloading spectre/lipreading pretrained model..." &&
        cd external/spectre &&
        gdown 1yHd4QwC7K_9Ro2OM_hC7pKUT2URPvm_f -O LRS3_V_WER32.3.zip &&
        unzip LRS3_V_WER32.3.zip -d data/ &&
        rm LRS3_V_WER32.3.zip
    )
)

# Link EMOCA and DECA configs
ln configs/emoca/cfg_baseline.yaml submodules/EMOCA/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg_baseline.yaml
ln configs/emoca/cfg_baseline.yaml submodules/EMOCA/assets/EMOCA/models/DECA/cfg_baseline.yaml
ln configs/emoca/cfg_spark.yaml submodules/EMOCA/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg_spark.yaml

# Setup SMIRK
(
    echo -e "\nDownloading pretrained SMIRK model..." &&
    cd submodules/SMIRK && 
    gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O pretrained_models/
    # Link SMIRK's FLAME to MultiFLARE's to avoid downloading it twice
    mkdir -p assets/FLAME2020 &&
    ln ../../../MultiFLARE/assets/flame/flame2020.pkl assets/FLAME2020/generic_model.pkl
)
