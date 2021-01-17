import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import os
from typing import List, NoReturn

def set_seed(seed: int, numpy_: bool=True, pytorch_: bool=True) -> NoReturn:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =  str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    