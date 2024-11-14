from olefile.olefile import OleFileError  # type: ignore[import-untyped]


class Txrm2TiffBaseException(Exception):
    pass


class Txrm2TiffIOError(Txrm2TiffBaseException, IOError):
    pass


class InvalidFileError(Txrm2TiffBaseException):
    pass


class TxrmError(Txrm2TiffBaseException):
    pass


class TxrmFileError(TxrmError, OleFileError):
    pass
