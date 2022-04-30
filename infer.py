# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
from PIL import Image

import paddle
import paddle.amp as amp
import paddle.nn.functional as F

import util.misc as misc
from util.datasets import build_transform

import models


def get_args_parser():
    parser = argparse.ArgumentParser('TokenLabeling training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='lvvit_t', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--infer_imgs', default='./demo/ILSVRC2012_val_00020010.JPEG', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--crop_pct', type=float, default=0.875)

    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser


def main(args):
    print("{}".format(args).replace(', ', ',\n'))

    preprocess = build_transform(is_train=False, args=args)

    model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes)

    misc.load_model(args, model)

    # switch to evaluation mode
    model.eval()

    infer_imgs = args.infer_imgs
    if isinstance(args.infer_imgs, str):
        infer_imgs = [args.infer_imgs]

    images = [Image.open(img).convert('RGB') for img in  infer_imgs]
    images = paddle.stack([preprocess(img) for img in images], axis=0)

    # compute output
    with amp.auto_cast():
        output = model(images)

    class_map = {}
    with open('demo/imagenet1k_label_list.txt', 'r') as f:
        for line in f.readlines():
            cat_id, *name = line.split('\n')[0].split(' ')
            class_map[int(cat_id)] = ', '.join(name)

    preds = []
    for file_name, scores, class_ids in zip(infer_imgs, *F.softmax(output).topk(5, 1)):
        preds.append({
            'class_ids': class_ids.tolist(),
            'scores': scores.tolist(),
            'file_name': file_name,
            'label_names': [class_map[i] for i in class_ids.tolist()]
        })

    print(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TokenLabeling training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.eval = True
    main(args)
