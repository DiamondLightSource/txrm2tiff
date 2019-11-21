# txrm2tiff

Converts txrm/xrm files to OME tiff files

## Instructions

**python run.py {input file path} {output file path (optional)}**

If no output path is supplied, the output tiff will be placed at the input path with the extension ".ome.tiff". The ".ome" signifies the OME xml metadata header.

If a reference has been applied within XMController, it will automatically apply the reference (image * 100 / reference). If it is a mosaic, this is recognised and the reference is tiled across the mosaic to be applied to each individual image.

Alternatively, dragndrop.bat has been supplied allowing windows users to drag and drop individual files for processing (note: you cannot set output path this way). This may require some setup depending on your Python installation, so please see the file.

## Requirements
* Python 3.*
* SciPy
* NumPy
* Tifffile
* olefile
