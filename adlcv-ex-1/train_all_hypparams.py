import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm

from text_classification import set_seed, main





if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()