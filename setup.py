import os
import sys
import setuptools

from src.txrm2tiff import __version__, __author__
from src.txrm2tiff.shortcut_creator import dragndrop_bat_file

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements=[
    "tifffile>=2020.9.30",
    "numpy>=1.17.4",
    "omexml-dls>=1.0.3",
    "olefile>=0.46",
    "scipy>=1.3.3",
    "pillow>=5.3",
    "pywin32;platform_system=='Windows'",
    ]

if os.name == "nt":
    batch_script_string = f"""{sys.executable} -m txrm2tiff --input %*
    """

    if dragndrop_bat_file.exists():
        dragndrop_bat_file.unlink()
    with open(dragndrop_bat_file, "x") as f:
        f.write(batch_script_string)

setuptools.setup(
    name="txrm2tiff",
    version=__version__,
    author=__author__,
    author_email="thomas.fish@diamond.ac.uk",
    description="A converter for Zeiss txrm and xrm files, created from B24 of Diamond Light Source",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    license_files=["LICENSE", os.path.join('font', 'License.txt')],
    url="https://github.com/DiamondLightSource/txrm2tiff",
    install_requires=requirements,
    packages=setuptools.find_packages('src', exclude=['scripts']),
    package_dir={'': 'src'},
    package_data={'': [os.path.join('font', 'CallingCode-Regular.otf')]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>3.6',
    entry_points={
        'console_scripts': [
            "txrm2tiff = txrm2tiff.scripts.commandline:main"
            ]
            },
    test_suite='tests',
    tests_require=['parameterized']
)
