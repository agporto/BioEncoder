# from .vis import *
from .core import utils
# from .scripts import *

def archive(*args, **kwargs):
    from .scripts.archive import archive as _archive
    return _archive(*args, **kwargs)


def configure(*args, **kwargs):
    from .scripts.configure import configure as _configure
    return _configure(*args, **kwargs)


def split_dataset(*args, **kwargs):
    from .scripts.split_dataset import split_dataset as _split_dataset
    return _split_dataset(*args, **kwargs)


def train(*args, **kwargs):
    from .scripts.train import train as _train
    return _train(*args, **kwargs)


def swa(*args, **kwargs):
    from .scripts.swa import swa as _swa
    return _swa(*args, **kwargs)


def lr_finder(*args, **kwargs):
    from .scripts.lr_finder import lr_finder as _lr_finder
    return _lr_finder(*args, **kwargs)


def interactive_plots(*args, **kwargs):
    from .scripts.interactive_plots import interactive_plots as _interactive_plots
    return _interactive_plots(*args, **kwargs)


def inference(*args, **kwargs):
    from .scripts.inference import inference as _inference
    return _inference(*args, **kwargs)


def model_explorer(*args, **kwargs):
    from .scripts.model_explorer_wrapper import model_explorer_wrapper as _model_explorer
    return _model_explorer(*args, **kwargs)

from importlib.metadata import version
__version__ = version("bioencoder")