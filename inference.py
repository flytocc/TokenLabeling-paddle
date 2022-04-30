# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from os import path as osp

from paddle import inference
from paddle.inference import Config, create_predictor

voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")
coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
wider_classes = (
                "Male","longHair","sunglass","Hat","Tshiirt","longSleeve","formal",
                "shorts","jeans","longPants","skirt","faceMask", "logo","stripe")

class_dict = {
    "voc07": voc_classes,
    "coco": coco_classes,
    "wider": wider_classes,
}


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[int(h_start):int(h_end), int(w_start):int(w_end), :]
    return img


def preprocess(img):
    img = Image.open(img)
    img = np.array(img)
    mean = [0, 0, 0]
    std = [1, 1, 1]
    img = resize_short(img, 448)
    img = crop_image(img, 448, True)
    # bgr-> rgb && hwc->chw
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img[np.newaxis, :]


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("CycleMLP Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("--dataset", default='voc07', type=str)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".jpg"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def main():
    args = parse_args()

    model_name = 'CycleMLP_B1'
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args)

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)

    if args.enable_benchmark:
        num_warmup = 0

        # instantiate auto log
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name=model_name,
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=0 if args.use_gpu else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # auto log start
        if args.enable_benchmark:
            autolog.times.start()

        # Pre process batched input
        batched_inputs = [files[st_idx:ed_idx]]
        imgs = []
        deal_imgs_name = []
        for inp in batched_inputs[0]:
            deal_imgs_name.append(inp)
            precess_im = preprocess(inp)  # preprocess
            imgs.append(precess_im)
        imgs = np.concatenate(imgs)
        batched_inputs = [imgs]
        # get pre process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # run inference
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(batched_inputs[i].shape)
            input_tensor.copy_from_cpu(batched_inputs[i].copy())

        # do the inference
        predictor.run()

        # get inference process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # get out data from output tensor
        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        result = results[0]
        for idx in range(result.shape[0]):
            print(deal_imgs_name[idx] + '\t' + 'prediction: ')
            logit = result[idx]
            pos = []
            logit = 1 / (1 + np.exp(-logit))  # sigmoid
            for i in range(len(logit)):
                if logit[i] > 0.5:
                    pos.append(i)
            for k in pos:
                print(class_dict[args.dataset][k], end=",")
            print()
        # get post process time cost
        if args.enable_benchmark:
            autolog.times.end(stamp=True)

    # report benchmark log if enabled
    if args.enable_benchmark:
        autolog.report()


if __name__ == "__main__":
    main()
