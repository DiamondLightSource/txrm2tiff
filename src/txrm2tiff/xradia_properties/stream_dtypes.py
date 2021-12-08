from .enums import XrmDataTypes


image_info_dict = {
    "ImageInfo/Angles": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/CameraBinning": XrmDataTypes.XRM_INT,
    "ImageInfo/CameraName": XrmDataTypes.XRM_STRING,
    "ImageInfo/CameraNumberOfFramesPerImage": XrmDataTypes.XRM_INT,
    "ImageInfo/CameraTemperature": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/CamPixelSize": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/Current": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/Date": XrmDataTypes.XRM_STRING,
    "ImageInfo/Energy": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/ExpTime": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/ExpTimes": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/HorizontalBin": XrmDataTypes.XRM_INT,
    "ImageInfo/ImageHeight": XrmDataTypes.XRM_INT,
    "ImageInfo/ImageWidth": XrmDataTypes.XRM_INT,
    "ImageInfo/ImagesPerProjection": XrmDataTypes.XRM_INT,
    "ImageInfo/ImagesTaken": XrmDataTypes.XRM_UNSIGNED_INT,
    "ImageInfo/IonChamberCurrent": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/MosaicFastAxis": XrmDataTypes.XRM_INT,
    "ImageInfo/MosaicSlowAxis": XrmDataTypes.XRM_INT,
    "ImageInfo/MosiacColumns": XrmDataTypes.XRM_INT,
    "ImageInfo/MosiacRows": XrmDataTypes.XRM_INT,
    "ImageInfo/MosiacMode": XrmDataTypes.XRM_INT,
    "ImageInfo/NoOfImages": XrmDataTypes.XRM_INT,
    "ImageInfo/ObjectiveName": XrmDataTypes.XRM_STRING,
    "ImageInfo/OpticalMagnification": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/PixelSize": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/Temperature": XrmDataTypes.XRM_INT,
    "ImageInfo/XPosition": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/XrayCurrent": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/XrayMagnification": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/YPosition": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/ZPosition": XrmDataTypes.XRM_FLOAT,
    "ImageInfo/ZonePlateName": XrmDataTypes.XRM_STRING,
}

reference_data_dict = {
    "ReferenceData/ExpTime": XrmDataTypes.XRM_FLOAT
    }

ref_image_info_dict = {f"ReferenceData/{k}": v for k, v in image_info_dict.items()}

dtypes_dict = {
    "ImageInfo/DataType": XrmDataTypes.XRM_INT,
    "ReferenceData/ImageInfo/DataType": XrmDataTypes.XRM_INT,
    "ReferenceData/DataType": XrmDataTypes.XRM_INT,
}
annotations_dict = {
    "Annot/TotalAnn": XrmDataTypes.XRM_INT,
}

position_info_dict = {
    "PositionInfo/AxisNames": XrmDataTypes.XRM_STRING,
    "PositionInfo/AxisUnits": XrmDataTypes.XRM_STRING,
    "PositionInfo/MotorPositions": XrmDataTypes.XRM_FLOAT,
    "PositionInfo/TotalAxis": XrmDataTypes.XRM_INT,
}

alignment_dict = {
    "Alignment/EncoderShiftsApplied": XrmDataTypes.XRM_INT,
    "Alignment/MetrologyShiftsApplied": XrmDataTypes.XRM_INT,
    "Alignment/ReferenceShiftsApplied": XrmDataTypes.XRM_INT,
    "Alignment/StageShiftsApplied": XrmDataTypes.XRM_INT,
    "Alignment/StaticRunoutApplied": XrmDataTypes.XRM_INT,
    "Alignment/X-Shifts": XrmDataTypes.XRM_FLOAT, # Pixels
    "Alignment/Y-Shifts": XrmDataTypes.XRM_FLOAT, # Pixels
}

misc = {"exeVersion": XrmDataTypes.XRM_STRING}

streams_dict = {}
streams_dict.update(image_info_dict)
streams_dict.update(reference_data_dict)
streams_dict.update(ref_image_info_dict)
streams_dict.update(dtypes_dict)
streams_dict.update(annotations_dict)
streams_dict.update(position_info_dict)
streams_dict.update(alignment_dict)
streams_dict.update(misc)
