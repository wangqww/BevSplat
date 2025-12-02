# BevSplat: Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026_Spotlight-red.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/118781)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09080-b31b1b.svg)](https://arxiv.org/abs/2502.09080)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation of BevSplat - Accepted as NeurIPS 2026 Spotlight**

This repository contains the code for our paper:
> **[Paper Title]** (Paper URL: https://arxiv.org/abs/2502.09080)

> **Abstract:** [Please provide your paper abstract here]

## 🌟 Key Features

- **Gaussian Splatting for 3D Scene Representation**: Novel approach to representing 3D scenes using Gaussian splatting techniques
- **Bird's Eye View (BEV) Generation**: Efficient conversion of panoramic images to bird's eye view for autonomous driving applications
- **Multi-Dataset Support**: Compatible with KITTI and VIGOR datasets
- **Robust Feature Extraction**: Advanced backbone networks including DINOv2, ResNet, and custom architectures
- **Panoramic Image Processing**: Specialized modules for handling panoramic imagery

## 📋 TODO - Coming Soon (Within 1 Week)

- [ ] **Open Source Dataset** - Release of pre-processed training and testing datasets
- [ ] **Pre-trained Weights** - Release of trained model weights for easy reproduction
- [ ] **Installation Scripts** - Automated environment setup
- [ ] **Web Demo** - Interactive online demonstration

*⚠️ Note: Dataset and model weights are currently being prepared for release and will be available within one week.*

## 🏗️ Repository Structure

```
BevSplat/
├── dataLoader/              # Dataset loading utilities
│   ├── KITTI_dataset.py     # KITTI dataset handling
│   ├── Oxford_dataset.py    # Oxford RobotCar dataset
│   ├── VIGOR_dataset_gs.py  # VIGOR dataset with Gaussian splatting
│   └── utils.py             # Data loading utilities
├── backbone/                # Feature extraction backbones
│   ├── backbone_dino.py     # DINO-based features
│   ├── backbone_resnet.py   # ResNet backbones
│   └── backbone_pano.py     # Panoramic-specific backbones
├── gaussian/                # Gaussian splatting modules
│   ├── encoder_feat.py      # Feature encoders
│   ├── pano_splat.py        # Panoramic splatting
│   └── build_gaussians.py   # Gaussian construction utilities
├── models/                  # Core model implementations
│   ├── bev_projection.py    # BEV projection modules
│   ├── dino.py              # DINO model variants
│   ├── dune.py              # DUNE model adaptations
│   └── feature_extractor.py # Feature extraction utilities
├── pano-gaussian_feat/      # Panoramic Gaussian Features
├── feat_gaussian/           # Feature-based Gaussian Splatting
├── train_KITTI_*.py         # Training scripts for different configurations
├── vis_gaussian_*.py        # Visualization utilities
└── transformer.py           # Transformer modules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU acceleration)
- OpenCV
- NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/BevSplat.git
cd BevSplat
```

2. Install dependencies:
```bash
# For panoramic Gaussian features
cd pano-gaussian_feat
pip install -e .

# For feature-based Gaussian splatting
cd ../feat_gaussian
pip install -e .
```

3. Install additional requirements:
```bash
pip install -r requirements.txt  # (Coming soon)
```

## 📊 Supported Datasets

### KITTI Dataset
- **Raw Data**: [KITTI Raw Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- **Odometry**: [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- **Weather Variants**: Support for different weather conditions

### Oxford RobotCar Dataset
- **Main Dataset**: [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)
- **Panoramic Processing**: Specialized loaders for Oxford panoramic imagery

### VIGOR Dataset
- **Cross-View Dataset**: [VIGOR](https://github.com/Jeffwang087/VIGOR)
- **Cross-City Generalization**: Support for cross-domain experiments

## 🏋️ Training

### KITTI Sequential Training
```bash
python train_KITTI_weak_seq.py \
    --dataset_path /path/to/KITTI \
    --batch_size 8 \
    --learning_rate 1e-4 \
```

### VIGOR 2DoF Training
```bash
python train_vigor_2DoF.py \
    --dataset_path /path/to/VIGOR \
    --model_type bev \
    --batch_size 16
```

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{bevsplat2026,
  title={[BevSplat Paper Title]},
  author={[Author Names]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  url={https://arxiv.org/abs/2502.09080}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- KITTI dataset providers
- VIGOR dataset authors
- NeurIPS 2026 reviewers and committee