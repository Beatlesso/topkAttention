'''
利用python中提供的setuptools模块完成事先编译流程，将写有算子的C++文件，编译成为一个动态链接库（在Linux平台是一个.so后缀文件）
可以让python调用其中实现的函数功能。需要setup.py编写如下
'''

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    # 找到当前目录的绝对路径，并且转到 /src 文件夹下
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    # main_file 就是当前文件夹下的所有.cpp文件   其它类似，分别在 /cpu 和 /cuda 文件夹下
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    # 首先用 main_file 和 source_cpu， 默认cpu版本
    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # 如果cuda可用，编译cuda版本
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError('Cuda is not availabel')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    # 确定待编译文件，及编译函数
    ext_modules = [
        extension(
            "TopkAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="TopkAttention",  # 编译后的链接库名称
    version="1.0",
    author="Yicong Luo",
    url="https://github.com/Beatlesso/topkAttention",
    description="PyTorch Wrapper for CUDA Functions of TopkAttention",  # 描述
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),   # 确定待编译文件，及编译函数
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},   # 执行编译命令设置
)
