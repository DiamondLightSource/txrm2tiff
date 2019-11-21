# txrm2tiff

Converts txrm/xrm files to OME tiff files

## Instructions

**python run.py {input_file_path} {output_file_path (optional)}**

if not output path is supplied, it will output at the input path ".ome.tiff" extension.

If a reference has been applied within XMController, it will automatically apply the reference (image * 100 / reference). If it is a mosaic, this is recognised and the reference is tiled across the mosaic to be applied to each individual image.

## Requirements
* Python 3.*
* scipy
* numpy
* tifffile
* olefile
