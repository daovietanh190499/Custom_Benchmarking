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
import torch.cuda.profiler as profiler

def train_loop(model, model_func, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, iteration, logger, run_info, forward_info):
    for nbatch, data in enumerate(train_dataloader):

        with torch.cuda.amp.autocast(enabled=run_info.amp):
            result = model_func(model, data, forward_info)
            loss = loss_func(result)

        if run_info.warmup is not None:
            warmup(optim, run_info.warmup, iteration, run_info.learning_rate)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if run_info.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())
        iteration += 1

    return iteration


def benchmark_train_loop(model, model_func, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, iteration, logger, run_info, forward_info):
    start_time = None
    # tensor for results
    result_ = torch.zeros((1,)).cuda()
    for nbatch, data in enumerate(loop(train_dataloader, run_info.reset_data)):
        if iteration >= run_info.benchmark_warmup:
            torch.cuda.synchronize()
            start_time = time.time()
        if iteration == run_info.benchmark_warmup + 1 and run_info.profile:
            profiler.start()

        with torch.cuda.amp.autocast(enabled=run_info.amp):
            result = model_func(model, data, forward_info)
            loss = loss_func(result)
        
        if iteration == run_info.benchmark_warmup + 1 and run_info.profile:
            profiler.stop()

        if run_info.warmup is not None:
            warmup(optim, run_info.warmup, iteration, run_info.learning_rate)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if iteration >= run_info.benchmark_warmup + run_info.benchmark_iterations:
            break

        if iteration >= run_info.benchmark_warmup:
            torch.cuda.synchronize()
            logger.update(run_info.batch_size*run_info.N_gpu, time.time() - start_time)
        iteration += 1

    result_.data[0] = logger.print_result()
    if run_info.N_gpu > 1:
        torch.distributed.reduce(result_, 0)
    if run_info.local_rank == 0:
        print('Training performance = {} FPS'.format(float(result_.data[0])))
    return iteration


def loop(dataloader, reset=True):
    while True:
        for data in dataloader:
            yield data
        if reset:
            dataloader.reset()

def benchmark_inference_loop(model, model_func, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, iteration, logger, run_info, forward_info):
    assert run_info.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    model.eval()
    val_datas = loop(val_dataloader, False)

    for i in range(run_info.benchmark_warmup + run_info.benchmark_iterations):
        torch.cuda.synchronize()
        start_time = time.time()

        data = next(val_datas)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=run_info.amp):
                _ = model_func(model, data, forward_info)

        torch.cuda.synchronize()
        end_time = time.time()


        if i >= run_info.benchmark_warmup:
            logger.update(run_info.eval_batch_size, end_time - start_time)

    logger.print_result()

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)
    