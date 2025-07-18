import setuptools
from setuptools import find_packages


def _load_version():
    with open("diplomat/__init__.py", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("__version__"):
                return line.split("=")[-1].strip().strip('"')

    raise ValueError("No version found!")


def _get_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


ONNX_TF_DEPS = [
    "h5py",
    "tables",
    "tensorflow",
    "tf2onnx>=1.16.1",
    "onnx",
    "onnxruntime",
]


VARIANTS = {
    "gui": ["wxpython"],
    "sleap": [*ONNX_TF_DEPS],
    "dlc": [*ONNX_TF_DEPS],
    "all": ["wxpython", *ONNX_TF_DEPS],
    "test": ["pytest", "scipy"],
}

DEVICE_VARIANTS = {
    "onnxruntime": {
        "": ["onnxruntime>=1.19"],
        "-nvidia": [
            "onnxruntime-gpu>=1.19",
            "nvidia-cuda-nvrtc-cu12~=12.0",
            "nvidia-cuda-runtime-cu12~=12.0",
            "nvidia-cufft-cu12~=11.0",
            "nvidia-curand-cu12~=10.0",
            "nvidia-cudnn-cu12~=9.0",
        ],
    }
}


def _attach_device_variants(variants, device_variants):
    new_variants = {}

    for name, pkgs in variants.items():
        non_device_pkgs = [pkg for pkg in pkgs if pkg not in device_variants]
        device_pkgs = [pkg for pkg in pkgs if pkg in device_variants]
        extensions = set(
            ext
            for device_pkg in device_pkgs
            for ext in device_variants[device_pkg].keys()
        )
        extensions.add("")

        for ext in extensions:
            new_pkgs = non_device_pkgs.copy()
            for device_pkg in device_pkgs:
                if ext in device_variants[device_pkg]:
                    new_pkgs.extend(device_variants[device_pkg][ext])
            new_variants[f"{name}{ext}"] = new_pkgs

    return new_variants


setuptools.setup(
    name="diplomat-track",
    version=_load_version(),
    author="Isaac Robinson",
    author_email="isaac.k.robinson2000@gmail.com",
    description="Deep learning-based Identity Preserving Labeled-Object Multi-Animal Tracking.",
    long_description=_get_readme(),
    long_description_content_type="text/markdown",
    url="https://diplomattrack.org",
    packages=find_packages(include="diplomat.*"),
    python_requires=">=3.8",
    platforms="any",
    entry_points={"console_scripts": ["diplomat=diplomat._cli_runner:main"]},
    project_urls={
        "Homepage": "https://diplomattrack.org",
        "Documentation": "https://diplomat.readthedocs.io",
        "Source": "https://github.com/TravisWheelerLab/DIPLOMAT",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # TODO: License
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Typing :: Typed",
    ],
    install_requires=[
        "opencv-python-headless",
        "matplotlib",
        "typing_extensions>=3.8",
        "tqdm",
        "PyYAML",
        "pandas",
        "numpy",
        "numba",
    ],
    extras_require=_attach_device_variants(VARIANTS, DEVICE_VARIANTS),
)
