from olefile import OleFileError


class TxrmError(Exception):
    pass


class TxrmFileError(TxrmError, OleFileError):
    pass
