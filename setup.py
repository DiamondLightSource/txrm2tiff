import os
import setuptools
from pathlib import Path


with Path("src/txrm2tiff/info.py").open() as info_file:
    exec(info_file.read())

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "tifffile>=2020.9.30",
    "numpy>=1.20",
    "omexml-dls>=1.1.0",
    "olefile>=0.46",
    "scipy>=1.3.3",
    "pillow>=5.3",
    "pywin32;platform_system=='Windows'",
]

setuptools.setup(
    name="txrm2tiff",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="A converter for Zeiss txrm and xrm files, created from B24 of Diamond Light Source",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    license_files=["LICENSE", os.path.join("font", "License.txt")],
    url="https://github.com/DiamondLightSource/txrm2tiff",
    install_requires=requirements,
    packages=setuptools.find_packages("src", exclude=["scripts"]),
    package_dir={"": "src"},
    package_data={"": [os.path.join("font", "CallingCode-Regular.otf")]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">3.6",
    entry_points={"console_scripts": ["txrm2tiff = txrm2tiff.__main__:main"]},
    test_suite="tests",
    tests_require=["parameterized"],
)
