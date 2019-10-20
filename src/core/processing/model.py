import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
import codecs
import random


#we fix the seeds to get consistent results

SEED = 234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
