# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import numpy as np

def evaluate(model, model_func, post_process, eval_func, val_dataloader, eval_info, forward_info):
    if eval_info.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    model.eval()
    if not forward_info.no_cuda:
        model.cuda()
    ret = []
    start = time.time()

    # for idx, image_id in enumerate(coco.img_keys):
    for nbatch, data in enumerate(val_dataloader):
        print("Parsing batch: {}/{}".format(nbatch, len(val_dataloader)), end='\r')
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=eval_info.amp):
                # Get predictions
                result = model_func(model, data, forward_info)

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            ret += post_process(result, data, forward_info)

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if eval_info.distributed:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).cuda()
        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]

        torch.cuda.synchronize()
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())
        torch.cuda.synchronize()

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size - ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
        # Everyone exchanges (padded) results

        torch.cuda.synchronize()
        torch.distributed.all_gather(other_ret, ret_pad)
        torch.cuda.synchronize()

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
    else:
        # Otherwise full results are just our results
        final_results = ret

    if eval_info.local_rank == 0:
        print("")
        print("Predicting Ended, total time: {:.2f} s".format(time.time() - start))

    eval_res = eval_func(final_results, eval_info.local_rank, forward_info)

    # put your model in training mode back on
    model.train()

    return eval_res

