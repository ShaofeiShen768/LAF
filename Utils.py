import torch
from torch import optim
import torch.nn.functional as F
import random
import random
import os
import numpy as np
'''
Random settings
'''
def random_setting(seed = 1):
    # random seed setting
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
