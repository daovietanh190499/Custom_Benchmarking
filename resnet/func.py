import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import torch.distributed as dist

def model_func(model, data, forward_info):
    if not forward_info['is_inference']:
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        if not forward_info['no_cuda']:
            input, target = input.cuda(), target.cuda()
    else:
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        if not forward_info['no_cuda']:
            input, target = input.cuda(), target.cuda()
    
    output = model(input)
    return output, target

def post_process(result, data, forward_info):
    return

def eval_func(final_results, forward_info):
    return