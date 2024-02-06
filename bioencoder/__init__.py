# from . import vis
# from . import core
# from . import scripts

## this might be removed in the future
from .vis import *
from .core import *

from .scripts.configure import configure
from .scripts.split_dataset import split_dataset
from .scripts.train import train
from .scripts.swa import swa
from .scripts.learning_rate_finder import learning_rate_finder
from .scripts.interactive_plots import interactive_plots
from .scripts.model_explorer_wrapper import model_explorer_wrapper as model_explorer
