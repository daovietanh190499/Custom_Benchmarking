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

import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed
from torchsummary import summary

from logger import Logger, BenchLogger
import dllogger as DLLogger

from train import train_loop, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from evaluate import evaluate
import wandb

# Apex imports
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


#nsys profile --show-output=true --export sqlite -o /results/test python main.py --model SSD300 --batch-size 32 --mode benchmark-training --benchmark-warmup 100 --benchmark-iterations 200 --data /coco --profile
#docker run --rm -it --gpus=all --ipc=host -v /home/hpc/coco:/coco -v /home/hpc/results:/results nvidia_universal_benchmark
#docker run --rm -it --gpus device=0 --ipc=host -v /home/hpc/coco:/coco -v /home/hpc/results:/results nvidia_universal_benchmark
#torchrun --nproc_per_node=1 main.py --model Unet3D --batch-size 2 --eval-batch-size 1 --mode benchmark-training --benchmark-warmup 100 --benchmark-iterations 200 --data /data --json-summary /results/log_unet3d_a100_fp16.log

def make_parser():
    parser = ArgumentParser(description="GPU Custom Benchmark")
    parser.add_argument('--model', '-md', type=str, default='SSD300', required=True,
                        help='name of model')
    parser.add_argument('--data', '-d', type=str, default='/coco', required=True,
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=65,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '--bs', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--save', type=str, default=None,
                        help='save model checkpoints in the specified directory')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64],
                        help='epochs at which to evaluate')
    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')

    # Hyperparameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=2.6e-3,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005,
                        help='momentum argument for SGD optimizer')

    # Benchmark
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--benchmark-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--benchmark-warmup', type=int, default=20, metavar='N',
                        help='Number of warmup iterations for benchmarking')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument("--amp", dest='amp', action="store_true",
                        help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--no-amp", dest='amp', action="store_false",
                        help="Disable Automatic Mixed Precision (AMP).")
    parser.set_defaults(amp=True)
    parser.add_argument("--allow-tf32", dest='allow_tf32', action="store_true",
                        help="Allow TF32 computations on supported GPUs.")
    parser.add_argument("--no-allow-tf32", dest='allow_tf32', action="store_false",
                        help="Disable TF32 computations.")
    parser.set_defaults(allow_tf32=True)
    parser.add_argument('--log-interval', type=int, default=20,
                        help='Logging interval.')
    parser.add_argument('--json-summary', type=str, default=None,
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument('--tensorboard-log', type=str, dest='tensorboard_log', default=None,
                        help='If provided, the tensorboard log will be written to'
                             'the specified file.')

    # Distributed
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    # Profiling
    parser.add_argument('--profile', dest='profile', action="store_true",
                        help='Used for profiling GPU')
    parser.add_argument('--profile-type', type=str, dest='profile_type', default='tensorboard',
                        choices=['nsys', 'tensorboard'])
    parser.add_argument('--wandb', dest='wandb', action="store_true",
                        help='Used for WanDB Logging')

    return parser


def train(train_loop_func, logger, args):
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda

    # Setup multi-GPU if necessary
    args.distributed = False
    args.world_size = 1
    args.seed = np.random.randint(1e4) if not args.seed else args.seed
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    else:
        args.N_gpu = 1

    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)


    # SSD300
    #================================================================================================================================================================
    if args.model == "SSD300":
        from ssd.model import SSD300, ResNet, Loss
        from ssd.utils import dboxes300_coco, Encoder, tencent_trick, generate_mean_std
        from ssd.data import get_train_dataloader, get_val_dataset, get_val_dataloader, get_coco_ground_truth
        from ssd.func import model_func, post_process, eval_func

        mean, std = generate_mean_std()
        cocoGt = get_coco_ground_truth(args.data)
        val_dataset = get_val_dataset(args.data)
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)

        # MUST HAVE PARAMETER
        forward_info = {
            'is_inference': args.mode in [ 'evaluation', 'benchmark-inference'],
            'no_cuda': args.no_cuda,
            'data_layout':"channels_last",
            'mean': mean,
            'std': std,
            'cocoGt': cocoGt,
            'val_dataset': val_dataset,
            'encoder': encoder
        }

        train_dataloader = get_train_dataloader(data=args.data, 
                                                batch_size=args.batch_size, 
                                                local_rank=args.local_rank, 
                                                N_gpu=args.N_gpu, 
                                                amp=args.amp, 
                                                num_workers=args.num_workers, 
                                                local_seed=args.seed - 2**31)

        val_dataloader = get_val_dataloader(val_dataset, 
                                            distributed=args.distributed, 
                                            eval_batch_size=args.eval_batch_size, 
                                            num_workers=args.num_workers)

        model = SSD300(backbone=ResNet(backbone='resnet50',
                                        backbone_path=None,
                                        weights='IMAGENET1K_V2'))
        summary(model.cuda(), (3,300,300), device='cuda')

        loss_func = Loss(dboxes)

        optimizer = torch.optim.SGD(tencent_trick(model), lr=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

        scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    #================================================================================================================================================================

    # U-Net3D
    #================================================================================================================================================================
    elif args.model == "Unet3D":
        from unet3d.unet3d import Unet3D
        from unet3d.losses import DiceCELoss, DiceScore
        from unet3d.dataloader import get_data_loaders
        from unet3d.func import model_func, post_process, eval_func

        # MUST HAVE PARAMETER
        forward_info = {
            'val_input_shape': [128, 128, 128],
            'overlap': 0.5,
            'world_size': args.world_size,
            'ga_steps': 1,
            'is_inference': args.mode in [ 'evaluation', 'benchmark-inference'],
            'no_cuda': args.no_cuda,
            'score_fn': DiceScore(to_onehot_y=True, use_argmax=True, layout='NCDHW', include_background=False)
        }

        train_dataloader, val_dataloader = get_data_loaders(data_dir=args.data,
                                                            loader="pytorch",
                                                            input_shape=[128, 128, 128],
                                                            val_input_shape=forward_info['val_input_shape'],
                                                            layout='NCDHW',
                                                            oversampling=0.4,
                                                            seed=args.seed,
                                                            batch_size=args.batch_size,
                                                            eval_batch_size =  args.eval_batch_size,
                                                            num_workers=args.num_workers,
                                                            benchmark='benchmark' in args.mode,
                                                            num_shards=args.world_size, global_rank=args.local_rank)

        model = Unet3D(1, 3, normalization='instancenorm', activation='relu')

        loss_func = DiceCELoss(to_onehot_y=True, use_softmax=True, layout='NCDHW',
                            include_background=False)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True,
                        weight_decay=args.weight_decay)

        scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    #===================================================================================================================================================================
    start_epoch = 0
    iteration = 0

    if use_cuda:
        model.cuda()
        loss_func.cuda()

    if args.distributed:
        model = DDP(model)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(model.module if args.distributed else model, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    # common train/inference config for all types of model
    class RunInfo:
        amp= args.amp
        warmup =  args.warmup
        learning_rate =  args.learning_rate
        local_rank =  args.local_rank
        benchmark_warmup =  args.benchmark_warmup
        benchmark_iterations =  args.benchmark_iterations
        batch_size =  args.batch_size
        eval_batch_size =  args.eval_batch_size
        N_gpu = args.N_gpu
        distributed =  args.distributed
        reset_data = args.model == 'SSD300'
        profile = args.profile
        profile_type = args.profile_type
        tensorboard_log = args.tensorboard_log
    run_info = RunInfo()

    total_time = 0
    if args.mode == 'evaluation':
        acc = evaluate(model, model_func, post_process, eval_func, val_dataloader, run_info, forward_info)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        for epoch in range(start_epoch, args.epochs):
            start_epoch_time = time.time()
            
            if args.distributed and args.model=="Unet3D":
                train_dataloader.sampler.set_epoch(epoch)
            
            if args.profile and args.profile_type == "nsys":
                with torch.autograd.profiler.emit_nvtx():
                    iteration = train_loop_func(model, model_func, loss_func, scaler, epoch, optimizer, train_dataloader, val_dataloader, iteration, logger, run_info, forward_info)
            else:
                iteration = train_loop_func(model, model_func, loss_func, scaler, epoch, optimizer, train_dataloader, val_dataloader, iteration, logger, run_info, forward_info)
            
            if args.mode in ["training", "benchmark-training"]:
                scheduler.step()
            
            end_epoch_time = time.time() - start_epoch_time
            total_time += end_epoch_time

            if args.local_rank == 0:
                logger.update_epoch_time(epoch, end_epoch_time)

            if epoch in args.evaluation:
                acc = evaluate(model, model_func, post_process, eval_func, val_dataloader, run_info, forward_info)
                if args.local_rank == 0:
                    logger.update_epoch(epoch, acc)

            if args.save and args.local_rank == 0:
                print("saving model...")
                obj = {'epoch': epoch + 1,
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),}
                if args.distributed:
                    obj['model'] = model.module.state_dict()
                else:
                    obj['model'] = model.state_dict()
                os.makedirs(args.save, exist_ok=True)
                save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
                torch.save(obj, save_path)
                logger.log('model path', save_path)
            
            if run_info.reset_data:
                train_dataloader.reset()

        DLLogger.log((), { 'total time': total_time })
        logger.log_summary()


def log_params(logger, args):
    logger.log_params({
        "model": args.model,
        "dataset path": args.data,
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "eval batch size": args.eval_batch_size,
        "no cuda": args.no_cuda,
        "seed": args.seed,
        "checkpoint path": args.checkpoint,
        "mode": args.mode,
        "eval on epochs": args.evaluation,
        "lr decay epochs": args.multistep,
        "learning rate": args.learning_rate,
        "momentum": args.momentum,
        "weight decay": args.weight_decay,
        "lr warmup": args.warmup,
        "num workers": args.num_workers,
        "AMP": args.amp,
        "allow-tf32": args.allow_tf32,
        "precision": 'amp fp16' if args.amp else 'fp32',
        "tensorboard_log": args.tensorboard_log,
        "profile": args.profile,
        "profile_type": args.profile_type,
        "wandb": args.wandb
    })

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project="benchmark", config=args)
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = True

    # write json only on the main thread
    args.json_summary = args.json_summary if args.local_rank == 0 else None

    if args.mode == 'benchmark-training':
        train_loop_func = benchmark_train_loop
        logger = BenchLogger('Training benchmark', log_interval=args.log_interval,
                             json_output=args.json_summary)
        args.epochs = 1
    elif args.mode == 'benchmark-inference':
        train_loop_func = benchmark_inference_loop
        logger = BenchLogger('Inference benchmark', log_interval=args.log_interval,
                             json_output=args.json_summary)
        args.epochs = 1
    else:
        train_loop_func = train_loop
        logger = Logger('Training logger', log_interval=args.log_interval,
                        json_output=args.json_summary)

    log_params(logger, args)

    train(train_loop_func, logger, args)
