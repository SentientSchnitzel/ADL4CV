import numpy as np
import os
import json
from itertools import product
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from text_classification import set_seed, main, prepare_data_iter





if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)


    """
    embed_dim 128, 256, 1024
    num_heads 4, 16, 32
    num_layers 4, 8, 16
    pos_enc (fixed / learnable)
    """
    
    def set_nested_value(d, keys, value):
        """ Recursively ensure all keys exist and set the value """
        for key in keys[:-1]:
            d = d.setdefault(key, {})  # Ensure intermediate levels exist
        d[keys[-1]] = value

    history = {}
    embed_dims_l = [256, 512]
    num_heads_l = [4, 8]
    num_layers_l = [4, 8]
    pos_encs = ["fixed", "learnable"]

    total_runs = len(embed_dims_l) * len(num_heads_l) * len(num_layers_l) * len(pos_encs)
    train_iter, val_iter, test_iter = prepare_data_iter(sampled_ratio=0.2,
                                                            batch_size=32)
    
    for embed_dim, num_heads, num_layers, pos_enc in tqdm(product(embed_dims_l, num_heads_l, num_layers_l, pos_encs), 
                                                        total=total_runs, desc="Running Experiments"):
        print(f" \
              Running experiment with \
              embed_dim: {embed_dim}, \
              num_heads: {num_heads}, \
              num_layers: {num_layers}, \
              pos_enc: {pos_enc}")
        h, test_metrics = main(embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               num_epochs=16,
                               pos_enc=pos_enc,
                               dropout=0.0,
                               lr=5e-4,
                               train_iter=train_iter,
                               val_iter=val_iter,
                               test_iter=test_iter,
                               )

        full_metrics = {**h, **test_metrics}
        set_nested_value(history, 
                        [f'ED {embed_dim}', 
                        f'NH {num_heads}', 
                        f'NL {num_layers}', 
                        f'PE {pos_enc}'], 
                        full_metrics)
        # print('set nested value')


    # for embed_dim in embed_dims_l:
    #     for num_heads in num_heads_l:
    #         for num_layers in num_heads_l:
    #             for pos_enc in pos_encs:
    #                 h = main(num_epochs=1)
    #                 set_nested_value(history, 
    #                                  [f'embed_dim: {embed_dim}', 
    #                                   f'num_heads: {num_heads}', 
    #                                   f'num_layers: {num_layers}', 
    #                                   f'pos_enc: {pos_enc}'], 
    #                                   h)
    
    json.dump(history, open('history_ex1.json', 'w'))
