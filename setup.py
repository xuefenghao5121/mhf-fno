from setuptools import setup, find_packages

setup(
    name="mhf-fno",
    version="1.0.0",
    description="Multi-Head Fourier Neural Operator Plugin for NeuralOperator",
    author="Tianyuan Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "neuraloperator>=2.0.0",
    ],
    python_requires=">=3.8",
)