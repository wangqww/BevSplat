#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import torch

_nvcc_flags = ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
if os.name == "nt":
    _cuda_ver = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2] if x.isdigit())
    if _cuda_ver >= (12, 0):
        _nvcc_flags += ["-Xcompiler", "/Zc:preprocessor"]

setup(
    name="pano_gaussian_feat",
    packages=['pano_gaussian_feat'],
    ext_modules=[
        CUDAExtension(
            name="pano_gaussian_feat._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": _nvcc_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
