# txrm2tiff

Converts txrm/xrm files to OME tif/tiff files

## Instructions

**python run.py {input file path} {output file path (optional, default=None)} {apply reference if available (optional, default=True)}**

e.g. _python run.py input.txrm_ to get the default output and apply a reference, if available
e.g. _python run.py input.xrm custom-output.ome.tif_ to get custom output and apply a reference, if available
e.g. _python run.py input.txrm None False_ to get the default output and ignore the reference

If no output path is supplied, the output file will be placed at the input path with the extension ".ome.tif"/".ome.tiff" as appropriate. The ".ome" signifies the OME XML metadata header.

**dragndrop.bat** has been supplied allowing windows users to drag and drop individual files for processing (note: you cannot set output path this way). This may require some setup depending on your Python installation, so please see the file.


## Features
* xrm/txrm files will be converted to tif/tiff.
* If a reference has been applied within XMController, it will automatically apply the reference (image * 100.0 / reference, as in XMController).
* If it is a mosaic, this is recognised and the reference will be applied to each individual image within the mosaic.
* Additional metadata will be added in OME XML format to the header.


## Requirements
* Python 3.*
* SciPy
* NumPy
* Tifffile
* olefile
