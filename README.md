# txrm2tiff

Converts TXRM/XRM files to OME-TIFF files.

Txrm2tiff was created for users of beamline B24 of Diamond Light Source by Thomas Fish. This has been adapted from, and is used by, B24's the automatic processing pipeline. Parts of this code were originally written by Kevin Savage, with further additions and amendments by Peter Chang, Victoria Beilsten-Edmands.

## Installation

Available on PyPI and conda-forge as `txrm2tiff`. To install:
- PyPI: `python -m pip install txrm2tiff`
- conda-forge: `conda install -c conda-forge txrm2tiff`

### Instructions

**txrm2tiff** {**--input** input file path (required)} {**--reference** reference file path (optional, default=None)} {**--output** path (optional, default=None)} {**--annotate** (optional)} {**--datetype** output data type (optional, choices=[uint16, float32, float64], default=None)} {**--ignore-ref** (optional)} {**--set-logging** (optional, default="info"}

**txrm2tiff -h** or **txrm2tiff --help** will give more info.
&nbsp;

##### Setup options:
**txrm2tiff setup** {**--windows-shortcut** (WINDOWS ONLY: optional, creates shortcut on the desktop for drag'n'drop processing)}

**txrm2tiff setup -h** or **txrm2tiff setup --help** will give more info.
&nbsp;

##### Inspector options:
**txrm2tiff inspect** {**--input** input file path (required)} {**--extra** (optional, default=False)} {**--list-streams** (optional, default=False)} {**--inspect-streams** space separated streams (optional)}

**txrm2tiff inspect --input** or **txrm2tiff inspect -i** followed by the path of a txrm/xrm file will output some basic information about the images contained.
  - Adding **--extra** or **-e** will add further information to this output.
  - Adding **--list-streams** or **-l** will list all of the streams\* in addition to the any previous output. This will be a lot so it may be useful to save this to a file using ` > file.txt`.
  - Adding **--inspect-streams** or **-s** followed by a 1 or more space separated stream names will read each stream using a variety of formats. As txrm and xrm files do not save the streams with information on what data type is being used\*\*, the output will take some interpreting.

**txrm2tiff inspect -h** or **txrm2tiff inspect --help** will give more info.


\*XRM and TXRM files are '[OLE](https://en.wikipedia.org/wiki/Object_Linking_and_Embedding)' type files. These files to separate and store information in streams.

\*\* with the exception of images, which do get the image data type saved separately.

**NOTE:** any commands beginning with `txrm2tiff` are essentially equivalent to usng `python -m txrm2tiff` (arguments will be parsed by the same parser via either method). This may be useful if there were any installation issues.

---

If no output path is supplied, the output file will be placed at the input path with the extension ".ome.tif"/".ome.tiff" as appropriate. The ".ome" signifies the OME XML metadata header.

**dragndrop.bat** has been supplied allowing windows users to drag and drop individual files or entire directories for processing (note: you cannot set output path this way). This may require some setup depending on your Python installation, so please see the file.

##### Logging options are:
* debug OR 1
* info OR 2
* warning OR 3
* error OR 4
* critical OR 5


### Examples:
**`txrm2tiff -h` and `txrm2tiff setup --h` will give more info**

`txrm2tiff -i input.txrm`
Saves "input.ome.tiff" with reference applied, if available.

`txrm2tiff -i input.txrm -r ref_stack.txrm`
Saves "input.ome.tiff" with a reference image applied from running Despeckle \& Average on the the txrm stack. This Despeckle \& Average algorithm is near-identical to the Zeiss algorithm, based on their logic.

`txrm2tiff --input input.txrm --reference ref_single.xrm --ignore-ref`
Saves "input.ome.tiff" with custom reference applied from a single image (e.g. a Despeckled_Ave.xrm file). If a custom reference is supplied, the ignore reference argument will be ignored.

`txrm2tiff -i input.xrm -o custom-output.ome.tif`
Saves "custom-output.ome.tif" with reference applied, if available.

`txrm2tiff -i input.xrm --annotate`
Saves "input.ome.tiff", as well as a separate file "input_Annotated.tif", which has annotations overlaid (if annotations are found) and scale bar.

`txrm2tiff --input input.xrm --ignore-ref --set-logging debug`
Saves "input.ome.tiff" and ignores any reference, shows debug and above level log messages.

`txrm2tiff -i input.xrm --output custom-output.ome.tif --set-logging error`
Saves "custom-output.ome.tiff", shows error and above level log messages.

**To batch convert:**
`txrm2tiff --input path/to/inputDirectory/`
Converts all XRM/TXRM files within input_directory with reference applied, if available.

`txrm2tiff --input path/to/inputDirectory/ --ignore-ref`
Converts all XRM/TXRM files within input_directory, ignoring all references.

`txrm2tiff --input path/to/inputDirectory/ --output path/to/outputDirectory/ --ignore-ref`
Converts all XRM/TXRM files within "inputDirectory", saving to the automatic name within the specified output directory, ignoring all references.

Batch conversion notes:
* `--output` _must_ be a directory or it will be ignored and files placed in the same directory as the XRM/TXRMs
* Sub directories containing any XRM/TXRM files found within "inputDirectory" will be copied to "outputDirectory" (directories will be created if they don't already exist)
* `--reference` inputs will be ignored for batch conversion


## Features
* Converts XRM/TXRM v3.0 (from XMController) & v5.0 (from XRMDataExporer) files to TIFF.
* If a reference has been applied within XMController/XRMDataExplorer, it will automatically apply the reference (_image * 100.0 / reference_, as done by XMController). Internal references can also be ignored.
* Custom reference images can be specified (can be a TXRM or XRM file, or a TIFF image or stack) - this option overrides any internally stored reference and the ignore reference option.
* If the reference exposure is available (e.g. from XRM/TXRM or OME-TIFF), the reference image will be rescaled to the image exposure at 0° (if applicable).
* Any annotations from XRM/TXRM v5.0 files can be exported and saved (along with a scale bar).
* Data type of the output image can be specified (warnings will be given if the data type limits off the dynamic range of the image, also values are rounded before casting float -> integer).
* Metadata will be added in OME XML format to the header.
* Batch convert options.
* Inspector (can extract any information from XRM/TXRM files).
* Within Python, XRM/TXRM files can be opened and interacted with using the function `open_txrm` (`from txrm2tiff import open_txrm`), which returns a `Txrm` object of the correct version. Recomended usage: `with open_txrm(...) as txrm:`.
* Within Python, XRM/TXRM files can quickly be converted and saved using `convert_and_save` (`from txrm2tiff import convert_and_save`).
