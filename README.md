<p align="center">
    <h1 align="center">SPARK: Self-supervised Personalized Real-time Monocular Face Capture</h1>
    <p align="center">
        <a href="mailto://kelian.baert@technicolor.com"><strong>Kelian Baert</strong></a>
        .
        <a href="https://sbharadwajj.github.io/"><strong>Shrisha Bharadwaj</strong></a>
        ·
        <a href="https://www.linkedin.com/in/fabien-castan/"><strong>Fabien Castan</strong></a>
        ·
        <a href="https://www.linkedin.com/in/benoitmaujean/"><strong>Benoit Maujean</strong></a>
        .
        <a href="https://people.irisa.fr/Marc.Christie/"><strong>Marc Christie</strong></a>
        ·
        <a href="https://vabrevaya.github.io/"><strong>Victoria Fernandez Abrevaya</strong></a>
        .
        <a href="https://boukhayma.github.io/"><strong>Adnane Boukhayma</strong></a>
    </p>
    <p align="center">
        <a href="https://technicolor.com">Technicolor</a> | <a href="https://is.mpg.de/">Max Planck Institute</a> | <a href="https://www.inria.fr/en/inria-centre-rennes-university">INRIA Rennes</a>
        <br>
        <strong>SIGGRAPH Asia 2024 Conference Papers</strong>
    </p>
    <p align="center">
        <a href="https://kelianb.github.io/SPARK/" style="padding-left: 0.5rem;">
            <img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue" alt="Project Page">
        </a>
        <a href="https://dl.acm.org/doi/10.1145/3680528">
            <img src="https://img.shields.io/badge/Paper-red" alt="PDF">
        </a>
        <a href="https://arxiv.org/abs/2409.07984">
            <img src="https://img.shields.io/badge/Arxiv-red" alt="arxiv PDF">
        </a>
    </p>
</p>
<p float="center">
    <img src="assets/teaser.gif" width="98%" />
</p>


## Citation

If you find our code or paper useful, please cite as:

```
@inproceedings{baert2024spark,
  title = {{SPARK}: Self-supervised Personalized Real-time Monocular Face Capture},
  author = {Baert, Kelian and Bharadwaj, Shrisha and Castan, Fabien and Maujean, Benoit and Christie, Marc and Abrevaya, Victoria and Boukhayma, Adnane},
  booktitle = {SIGGRAPH Asia 2024 Conference Proceedings},
  doi = {10.1145/3680528.3687704},
  isbn = {979-8-4007-1131-2/24/12},
  month = dec,
  year = {2024},
  url = {https://kelianb.github.io/SPARK/},
}
```

## Installation

<details>
    <summary>Details</summary>

- Create the environment using [setup.sh](./setup.sh).
- Run [TrackerAdaptation/setup_submodules.sh](./TrackerAdaptation/setup_submodules.sh). This may take a few minutes.
- Link FLAME from MultiFLARE to EMOCA: `ln TrackerAdaptation/submodules/EMOCA/assets/FLAME/geometry/generic_model.pkl MultiFLARE/assets/flame/flame2020.pkl`
    - This is equivalent to downloading [FLAME](https://flame.is.tue.mpg.de/download.php) (2020 version), unzipping it and copying `generic_model.pkl` at `./MultiFLARE/assets/flame/flame2020.pkl`.
- Get Basel Face Model texture space adapted to FLAME. Unfortunately, we are not allowed to distribute the texture space since the license does not permit it. Therefore, please use the tool from this [repo](https://github.com/TimoBolkart/BFM_to_FLAME) to convert the texture space to FLAME. Put the resulting texture model file at `TrackerAdaptation/submodules/EMOCA/assets/FLAME/texture/FLAME_albedo_from_BFM.npz`.

SPARK has been tested with NVIDIA RTX A5000 (24GB) or RTX A4000 (16GB) GPUs. It is possible to train on GPUs with less memory by reducing the batch size. 

</details>

## Dataset

Please refer to the [MonoFaceCompute](https://github.com/KelianB/MonoFaceCompute) repository to preprocess your own data.

## Usage

SPARK is a two-stage approach. First, run [MultiFLARE](./MultiFLARE/) to reconstruct a 3D Face Avatar from multiple videos. Then, use [TrackerAdaptation](./TrackerAdaptation/) to adapt an existing 3D face tracker to your avatar for real-time tracking through transfer learning. 

<details>
    <summary>Details</summary>

### 1. MultiFLARE
```bash
cd MultiFLARE
python train.py --config configs/example.txt

# Export neutral mesh
python export_mesh.py --config configs/example.txt --resume 3000 --out_dir /tmp/example_mesh --tex_type albedo
```

We advise starting from the provided example config and modifying `input_dir`, `train_dir` and `output_dir`. For a list of all parameters, please refer to [arguments.py](./MultiFLARE/arguments.py) or the output of `python train.py --help`. Parameters can be passed either in the config file or as command line arguments.

### 2. TrackerAdaptation
```bash
cd TrackerAdaptation
# DECA encoder + MultiFLARE decoder
python train.py --config configs/example_deca.txt
# EMOCA encoder + MultiFLARE decoder
python train.py --config configs/example_emoca.txt
# SMIRK encoder + MultiFLARE decoder (recommended!)
python train.py --config configs/example_smirk.txt
# EMOCA encoder + EMOCA decoder (baseline)
python train.py --config configs/example_emoca_baseline.txt

# Quantitative eval
python evaluate.py --config configs/example_smirk.txt --tracker_resume 3000 --frame_interval 5 --num_frames 64

# Visualization videos
python make_comparison_video.py --config configs/example_smirk.txt --tracker_resume 3000 --test_dirs 5 6 --n_frames 1000 --smooth_crops --framerate 24
python make_overlay_video.py --config configs/example_smirk.txt --tracker_resume 3000 --test_dirs 2 --out test_beard --texture /path/to/texture.png --n_frames 1000 --smooth_crops --framerate 24
```

</details>

## License Information

The code in this repository is subject to multiple licenses. 

1. **Original Code** (Technicolor Group & INRIA Rennes)
   - All code in this repository, except where otherwise specified, is licensed under the [CC BY-NC-SA License](./LICENSE).

2. **Third-Party Code** (Max Planck Institute for Intelligent Systems)
   - Location: `./MultiFLARE/flame`, `./MultiFLARE/flare`, `./TrackerAdaptation/submodules`
   - These directories contain code by Max Planck Institute, with some modifications. Please carefully read the [MPI License](./LICENSE_MPI) and note that this is only available for **non-commercial scientific research purposes**.
