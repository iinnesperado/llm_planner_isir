from setuptools import setup, find_packages

setup(
    name="object_reid_pillar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "open_clip_torch", 
        "torch",
        "opencv-python",
        "numpy<2.0",
        "tqdm"
    ],
)
