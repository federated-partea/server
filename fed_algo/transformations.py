import pickle


def serialize_bytes(p):
    return pickle.dumps(p)


def deserialize_bytes(p):
    return pickle.loads(p)


CELL_SEPARATOR = ','


def escape_cell(val):
    return val.replace(CELL_SEPARATOR, '\\SEP_CHAR')


def unescape_cell(val):
    return val.replace('\\SEP_CHAR', CELL_SEPARATOR)
