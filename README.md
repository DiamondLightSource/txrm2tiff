# txrm2tiff

Converts txrm/xrm files to OME tif/tiff files

## Instructions

**python txrm2tiff.py** {input file path (positional, required)} {**--reference-using** reference file path (optional, default=None)} {**--output** path (optional, default=None)} {**--ignore-ref** (optional, default=False)}

If no output path is supplied, the output file will be placed at the input path with the extension ".ome.tif"/".ome.tiff" as appropriate. The ".ome" signifies the OME XML metadata header.

**dragndrop.bat** has been supplied allowing windows users to drag and drop individual files for processing (note: you cannot set output path this way). This may require some setup depending on your Python installation, so please see the file.

Alternatively, the function **run(args)** can be accessed in "src/run.py" if you wish to use txrm2tiff in iPython. This uses the same inputs as above.

### Examples:
**python txrm2tiff.py input.txrm**
Saves "input.ome.tiff" with reference applied, if available.

**python txrm2tiff.py input.txrm ref_stack.txrm**
Saves "input.ome.tiff" with custom reference applied using the median of a txrm stack.

**python txrm2tiff.py input.txrm --output ref_single.xrm --ignore-ref**
Saves "input.ome.tiff" with custom reference applied from a single image (e.g. a Despeckled_Ave.xrm file). If a custom reference is supplied, the ignore reference argument will be ignored.

**python txrm2tiff.py input.xrm --output custom-output.ome.tif**
Saves "custom-output.ome.tif" with reference applied, if available.

**python txrm2tiff.py input.xrm --ignore-ref**
Saves "input.ome.tiff" and ignores any reference.


## Features
* xrm/txrm files will be converted to tif/tiff.
* If a reference has been applied within XMController, it will automatically apply the reference (_image * 100.0 / reference_, as done by XMController).
* Internally stored reference images can be ignored.
* A separate file containing reference images can be specified (either a txrm stack or a single xrm image) - this overrides any internally stored reference and the ignore reference option.
* If it is a mosaic, this is recognised and the reference will be applied to each individual image within the mosaic.
* Additional metadata will be added in OME XML format to the header.



## Requirements
Python 3.* (3.7 is recommended)
See "stable-requirements.txt" for modules

**pip install -r "stable-requirements.txt"** will install the required modules on your machine, if you do not already have them (use the "-u" decorator to install for the current user only).

### Setting up a virtual environment for txrm2tiff
Installing the required modules can be done within a virtual environment (venv) to avoid conflicting up other installations by using:
**python3 -m venv env** in the txrm2tiff directory.

To activate the environment, enter the command:
**source env/bin/activate** when in the txrm2tiff directory.

After this, the modules can be installed as shown above and they will only exist within this virtual environment. When set up this way, the venv will always need to be activated before running txrm2tiff.
