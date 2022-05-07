# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse

import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

import util.misc as misc
from util.datasets import build_dataset

from engine import evaluate

import models


def get_args_parser():
    parser = argparse.ArgumentParser('TokenLabeling training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='lvvit_t', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--use_amp', action='store_true', help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--no_use_amp', action='store_false', dest='use_amp')
    parser.set_defaults(use_amp=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--cls_label_path_val', default=None, type=str,
                        help='dataset label path val')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--crop_pct', type=float, default=0.875)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("{}".format(args).replace(', ', ',\n'))

    dataset_val = build_dataset(is_train=False, args=args)

    if args.dist_eval:
        num_tasks = misc.get_world_size()
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = DistributedBatchSampler(
            dataset_val, args.batch_size, shuffle=True, drop_last=False)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = BatchSampler(dataset=dataset_val, batch_size=args.batch_size)

    data_loader_val = DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=args.num_workers)

    model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes)

    model = paddle.DataParallel(model)
    model_without_ddp = model._layers
    n_parameters = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    print(f'number of params: {n_parameters / 1e6} M')

    misc.load_model(args, model_without_ddp)

    test_stats = evaluate(data_loader_val, model, use_amp=args.use_amp)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TokenLabeling training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.eval = True
    main(args)
