# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import random
import os
import numpy as np
import time
import json

from pathlib import Path

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

import util.misc as misc
from util.data import Mixup
from util.datasets import build_dataset
from util.loss import LabelSmoothingCrossEntropy, TokenLabelCrossEntropy
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import WandbLogger
from util.model_ema import ModelEma, unwrap_model
from engine import train_one_epoch, evaluate

import models


def get_args_parser():
    parser = argparse.ArgumentParser('TokenLabeling training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='lvvit_t', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')

    # Model Exponential Moving Average
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--no_model_ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_eval', action='store_true', help='Using ema to eval during training.')
    parser.add_argument('--no_model_ema_eval', action='store_false', dest='model_ema_eval')
    parser.set_defaults(model_ema_eval=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--use_amp', action='store_true', help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--no_use_amp', action='store_false', dest='use_amp')
    parser.set_defaults(use_amp=False)

    # Learning rate schedule parameters
    parser.add_argument('--blr', type=float, default=8e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 512')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--t_in_epochs', action='store_true')
    parser.add_argument('--no_t_in_epochs', action='store_false', dest='t_in_epochs')
    parser.set_defaults(t_in_epochs=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--crop_pct', type=float, default=0.875)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Token labeling
    parser.add_argument('--token_label', action='store_true', default=False,
                        help='Use dense token-level label map for training')
    parser.add_argument('--token_label_data', type=str, default='', metavar='DIR',
                        help='path to token_label data')
    parser.add_argument('--token_label_size', type=int, default=1, metavar='N',
                        help='size of result token label map')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--remove_head_from_pretained', action='store_true')
    parser.add_argument('--no_remove_head_from_pretained', action='store_false',
                        dest='remove_head_from_pretained')
    parser.set_defaults(remove_head_from_pretained=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--cls_label_path_train', default=None, type=str,
                        help='dataset label path train')
    parser.add_argument('--cls_label_path_val', default=None, type=str,
                        help='dataset label path val')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_file', default='',
                        help='log filename, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_ema', default='', help='resume ema from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    # logger training parameters
    parser.add_argument('--log_wandb', action='store_true',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='user or team name of wandb')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--debug', action='store_true')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    paddle.seed(args.seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.debug:
        paddle.version.cudnn.FLAGS_cudnn_deterministic = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    sampler_train = DistributedBatchSampler(
        dataset_train, args.batch_size, shuffle=True, drop_last=True)
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

    data_loader_train = DataLoader(dataset_train, batch_sampler=sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=args.num_workers)

    if misc.get_rank() == 0 and args.log_wandb and not args.eval:
        log_writer = WandbLogger(args, entity=args.wandb_entity, project=args.wandb_project)
    else:
        log_writer = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path)

    if args.finetune:
        checkpoint = paddle.load(args.finetune)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and (args.remove_head_from_pretained or checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.set_state_dict(checkpoint_model)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 512
        print("base lr: %.2e" % (args.lr * 512 / eff_batch_size))

    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.opt == 'adamw':
        skip_list = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
        decay_dict = {
            param.name: not (len(param.shape) == 1 or name.endswith(".bias") or name in skip_list)
            for name, param in model.named_parameters()}
        bete1, beta2 = args.opt_betas or (0.9, 0.999)
        optimizer = optim.AdamW(
            learning_rate=args.lr,
            beta1=bete1, beta2=beta2,
            epsilon=args.opt_eps,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda n: decay_dict[n],
            grad_clip=nn.ClipGradByGlobalNorm(args.clip_grad) if args.clip_grad is not None else None
        )
    elif args.opt == 'momentum':
        optimizer = optim.Momentum(
            learning_rate=args.lr,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=nn.ClipGradByGlobalNorm(args.clip_grad) if args.clip_grad is not None else None
        )
    loss_scaler = NativeScaler() if args.use_amp else None

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(model, decay=args.model_ema_decay, resume='')

    model = paddle.DataParallel(model)
    model_without_ddp = model._layers
    n_parameters = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    print(f'number of params: {n_parameters / 1e6} M')

    if args.token_label:
        if args.token_label_size == 1:
            # back to relabel/original ImageNet label
            raise NotImplementedError
        else:
            criterion = TokenLabelCrossEntropy()
    elif args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = nn.CrossEntropyLoss(soft_label=True)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    misc.load_model(args, model_without_ddp,
                    model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, use_amp=args.use_amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    total_epochs = args.epochs + args.cooldown_epochs
    print(f"Start training for {total_epochs} epochs")

    start_time = time.time()
    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    if args.resume:
        test_stats = evaluate(data_loader_val, model, use_amp=args.use_amp)
        max_accuracy = max(max_accuracy, test_stats['acc1'])
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if args.model_ema and args.model_ema_eval:
            test_stats_ema = evaluate(data_loader_val, unwrap_model(model_ema), use_amp=args.use_amp)
            max_accuracy_ema = max(max_accuracy_ema, test_stats_ema['acc1'])
            print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")

    for epoch in range(args.start_epoch, total_epochs):
        data_loader_train.batch_sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            model_ema, mixup_fn,
            log_writer=log_writer,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            use_amp=args.use_amp,
            args=args
        )
        test_stats = evaluate(data_loader_val, model, use_amp=args.use_amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir:
            misc.save_model(args, epoch, model_without_ddp,
                            model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='latest')
            if test_stats["acc1"] > max_accuracy:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='best')
            if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'best_acc1': max_accuracy,
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # repeat testing routines for EMA, if ema eval is turned on
        if args.model_ema and args.model_ema_eval:
            test_stats_ema = evaluate(data_loader_val, unwrap_model(model_ema), use_amp=args.use_amp)
            print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
            if args.output_dir and test_stats_ema["acc1"] > max_accuracy_ema:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='best-ema')
            max_accuracy_ema = max(max_accuracy_ema, test_stats_ema["acc1"])
            print(f'Max accuracy of the model EMA: {max_accuracy_ema:.2f}%')
            log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()},
                             'best_acc1_ema': max_accuracy_ema})

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.update(log_stats)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TokenLabeling training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
