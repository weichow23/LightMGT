from setuptools import setup, find_packages

setup(
    name="lightmgt",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "diffusers>=0.27.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "einops>=0.7.0",
    ],
)
