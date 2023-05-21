from common import config

if config.GPU:
    import cupy

    np = cupy
else:
    import numpy

    np = numpy
