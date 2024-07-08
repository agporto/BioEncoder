# from .vis import *
from .core import utils
# from .scripts import *

from .scripts.archive import archive
from .scripts.configure import configure
from .scripts.split_dataset import split_dataset
from .scripts.train import train
from .scripts.swa import swa
from .scripts.lr_finder import lr_finder
from .scripts.interactive_plots import interactive_plots
from .scripts.inference import inference
from .scripts.model_explorer_wrapper import model_explorer_wrapper as model_explorer

from importlib.metadata import version
__version__ = version("bioencoder")