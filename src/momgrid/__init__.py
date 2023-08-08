""" momgrid - flexible grid object for MOM-based Ocean Models """

import importlib.metadata as ilm

msg = ilm.metadata("momgrid")

__name__ = msg["Name"]
__version__ = msg["Version"]
__license__ = msg["License"]
__email__ = msg["Maintainer-email"]
__description__ = msg["Summary"]
__requires__ = msg["Requires-Dist"]
__requires_python__ = msg["Requires-Python"]

from . import util
from .classes import MOMgrid
