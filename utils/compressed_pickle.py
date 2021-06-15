#  Copyright (c) 2021 Dafne-Imaging Team

import pickle
import bz2

COMPRESS_LEVEL = 9


def compressed_dump(obj, file, **kwargs):
    with bz2.BZ2File(file, 'wb', compresslevel=COMPRESS_LEVEL) as f:
        pickle.dump(obj, f, **kwargs)


def compressed_dumps(obj, **kwargs):
    return bz2.compress(pickle.dumps(obj, **kwargs), compresslevel=COMPRESS_LEVEL)


def compressed_load(file, **kwargs):
    with bz2.BZ2File(file, 'rb') as f:
        return pickle.load(f, **kwargs)


def compressed_loads(compressed_bytes, **kwargs):
    return pickle.loads(bz2.decompress(compressed_bytes), **kwargs)


def loads(byte_array, **kwargs):
    """
    Generic loads to replace pickle load
    """
    try:
        return compressed_loads(byte_array, **kwargs)
    except OSError:
        print("Loading uncompressed pickle")
        return pickle.loads(byte_array, **kwargs)
    return None


def load(file, **kwargs):
    """
    Generic load
    """
    try:
        return compressed_load(file, **kwargs)
    except OSError:
        print("Loading uncompressed pickle")
        try:
            file.seek(0)
        except:
            pass
        return pickle.load(file, **kwargs)
    return None


dump = compressed_dump
dumps = compressed_dumps
