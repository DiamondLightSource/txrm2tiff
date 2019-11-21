# txrm2tiff

Converts txrm/xrm files to OME tiff files

## Instructions

**python run.py {input_file_path} {output_file_path (optional)}**

if not output path is supplied, it will output at the input path ".ome.tiff" extension.

If a reference has been applied within XMController, it will automatically apply the reference (image * 100 / reference). If it is a mosaic, this is recognised and the reference is tiled across the mosaic to be applied to each individual image.

Alternatively, dragndrop.bat has been supplied allowing windows users to drag and drop individual files for processing (note: you cannot set output path this way). This may require some setup depending on your Python installation, so please see the file.

## Requirements
* Python 3.*
* SciPy
* NumPy
* Tifffile
* olefile
