"""
MHF-FNO Package Setup

安装方式:
    pip install -e .  # 开发模式
    pip install .     # 安装模式
"""

from setuptools import setup, find_packages

# 读取 README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取 requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mhf-fno",
    version="1.4.0",
    author="Tianyuan Team",
    author_email="tianyuan@example.com",
    description="Multi-Head Fourier Neural Operator with Cross-Head Attention (CoDA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xuefenghao5121/mhf-fno",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "benchmark", "benchmark.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "benchmark": [
            "h5py>=3.0.0",
            "tensorboard>=2.0.0",
        ],
    },
    keywords="fourier neural operator deep learning pde spectral",
)