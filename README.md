# All Tokens Matter: Token Labeling for Training Better Vision Transformers

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. 代码结构](#5-代码结构)
* [6. 自动化测试脚本](#6-自动化测试脚本)
* [7. License](#7-license)
* [8. 参考链接与文献](#8-参考链接与文献)

## 1. 简介

这是一个PaddlePaddle实现的All Tokens Matter。

**论文:** [All Tokens Matter: Token Labeling for Training Better Vision Transformers](https://arxiv.org/abs/2104.10858)

**参考repo:** [TokenLabeling](https://github.com/zihangJiang/TokenLabeling)

在此非常感谢`zihangJiang`贡献的[TokenLabeling](https://github.com/zihangJiang/TokenLabeling)，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### 复现精度

您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。

| 模型      | top1 acc (参考精度) | top1 acc (复现精度) | 权重 \| 训练日志 |
|:---------:|:------:|:----------:|:----------:|
| lvvit_t | 0.791 (w/ AMP) | 0.796 (w/o AMP) | lvvit_t_ema.pd \| log.txt |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1UCU9zTrRo021lDbd60scpA?pwd=vpeg)


## 3. 准备数据与环境

### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * RTX3090
- 框架：
  - PaddlePaddle >= 2.2.0

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/TokenLabeling-paddle.git
cd TokenLabeling-paddle
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

如果您已经ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。


## 4. 开始使用


### 4.1 模型训练

* 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
    main.py \
    --model lvvit_t --drop_path 0.1 \
    --use_amp \
    --batch_size 128 \
    --warmup_epochs 5 --cooldown_epochs 10 \
    --lr 1.6e-3 --warmup_lr 1e-6 --min_lr 1e-5 --t_in_epochs \
    --mixup 0 --cutmix 0 --train_interpolation random \
    --token_label \
    --token_label_data /path/to/label_top5_train_nfnet \
    --token_label_size 14 \
    --data_path /path/to/imagenet/ \
    --cls_label_path_train /path/to/train_list.txt \
    --cls_label_path_val /path/to/val_list.txt \
    --output_dir output/lvvit_t/ \
    --crop_pct 0.9 --dist_eval
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_path`下train/val里的图片作为train-set/val-set。

部分训练日志如下所示。

```
[12:18:32.163049] Epoch: [279]  [1120/1251]  eta: 0:02:09  lr: 0.000030  loss: 10.1298 (9.9082)  time: 0.9781  data: 0.0084
[12:18:51.692836] Epoch: [279]  [1140/1251]  eta: 0:01:49  lr: 0.000030  loss: 9.9401 (9.9076)  time: 0.9762  data: 0.0119
```

### 4.2 模型评估

paddle bug: 不同batch_size得到的acc差别很大。

通过.pdparams转.pth在[TokenLabeling](https://github.com/zihangJiang/TokenLabeling)测出的acc1 79.638 (without AMP)。

``` shell
python eval.py \
    --model lvvit_t \
    --batch_size 512 \
    --crop_pct 0.9 \
    --data_path /path/to/imagenet/ \
    --cls_label_path_val /path/to/val_list.txt \
    --resume $TRAINED_MODEL
```

ps: 如果未指定`cls_label_path_val`，会读取`data_path`/val里的图片作为val-set。

### 4.3 模型预测

```shell
python predict.py \
    --model lvvit_t \
    --crop_pct 0.9 \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 171, 246, 211, 209], 'scores': [0.9884118437767029, 0.00022494762379210442, 0.00021465634927153587, 0.0001764898479450494, 0.00013726181350648403], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'Italian, greyhound', 'Great, Dane', 'vizsla,, Hungarian, pointer', 'Chesapeake, Bay, retriever']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.9884118437767029`。

### 4.4 模型导出

```shell
python export_model.py \
    --model lvvit_t \
    --output_dir /path/to/save/export_model/ \
    --resume $TRAINED_MODEL

python infer.py \
    --model_file /path/to/save/export_model/output/model.pdmodel \
    --params_file /path/to/save/export_model/output/model.pdiparams \
    --input_file ./demo/ILSVRC2012_val_00020010.JPEG
```

输出结果为
```
[{'class_ids': [178, 246, 171, 211, 209], 'scores': [0.9883329272270203, 0.00022819697915110737, 0.00021366256987676024, 0.0001897417096188292, 0.00013097828195896], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'Great Dane', 'Italian greyhound', 'vizsla, Hungarian pointer', 'Chesapeake Bay retriever']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.9883329272270203`。与predict.py结果的误差在正常范围内。

## 5. 代码结构

```
├── demo
├── engine.py
├── eval.py
├── export_model.py
├── infer.py
├── predict.py
├── main.py
├── models.py
├── README.md
├── requirements.txt
├── test_tipc
└── util
```


## 6. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [TIPC: test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/config/TokenLabeling/lvvit_t.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/config/TokenLabeling/lvvit_t.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
Run successfully with command - python3.7 main.py --model=lvvit_t --drop_path=0.1 --lr=1.6e-3 --warmup_lr=1e-6 --min_lr=1e-5 --t_in_epochs --mixup=0 --cutmix=0 --crop_pct=0.9 --train_interpolation=random --data_path=./dataset/ILSVRC2012/ --cls_label_path_train=./dataset/ILSVRC2012/train_list.txt --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --token_label --token_label_data=./dataset/ILSVRC2012/label_top5_train_nfnet/ --token_label_size=14 --dist_eval --output_dir=./test_tipc/output/norm_train_gpus_0_autocast_null/lvvit_t --epochs=2 --batch_size=8 !
Run successfully with command - python3.7 eval.py --model=lvvit_t --crop_pct=0.9 --data_path=./dataset/ILSVRC2012/ --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/lvvit_t/checkpoint-latest.pd !
Run successfully with command - python3.7 export_model.py --model=lvvit_t --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/lvvit_t/checkpoint-latest.pd --output=./test_tipc/output/norm_train_gpus_0_autocast_null !
Run successfully with command - python3.7 infer.py --use_gpu=True --use_tensorrt=False --precision=fp32 --model_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdmodel --batch_size=1 --input_file=./dataset/ILSVRC2012/val  --params_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdiparams > ./test_tipc/output/python_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log 2>&1 !
......
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。


## 7. License

All Tokens Matter is released under Apache-2.0 License.

## 8. 参考链接与文献
1. All Tokens Matter: Token Labeling for Training Better Vision Transformers: https://arxiv.org/abs/2104.10858
2. TokenLabeling: https://github.com/zihangJiang/TokenLabeling

再次感谢`zihangJiang`等人贡献的[TokenLabeling](https://github.com/zihangJiang/TokenLabeling)，提高了本repo复现论文的效率。

```
@inproceedings{NEURIPS2021_9a49a25d,
 author = {Jiang, Zi-Hang and Hou, Qibin and Yuan, Li and Zhou, Daquan and Shi, Yujun and Jin, Xiaojie and Wang, Anran and Feng, Jiashi},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {18590--18602},
 publisher = {Curran Associates, Inc.},
 title = {All Tokens Matter: Token Labeling for Training Better Vision Transformers},
 url = {https://proceedings.neurips.cc/paper/2021/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
