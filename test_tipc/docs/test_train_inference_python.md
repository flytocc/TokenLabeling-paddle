# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
| :----- | :------ | :------ | :----- |
| TokenLabeling | lvvit_t | 正常训练 | 正常训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
| :-------- | :------ | :------ | :--- | :------ |
| TokenLabeling | lvvit_t | 支持 | 支持 | 1 |


## 2. 测试流程

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 安装依赖
- 安装PaddlePaddle >= 2.2
- 安装PaddleClas依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装autolog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```

### 2.2 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。

`test_train_inference_python.sh`当前只包含1种运行模式：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash test_tipc/prepare.sh test_tipc/config/TokenLabeling/lvvit_t.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/config/TokenLabeling/lvvit_t.txt 'lite_train_lite_infer'
```

运行相应指令后，在`test_tipc/output`文件夹下自动会保存运行日志。如'lite_train_lite_infer'模式下，会运行训练+inference的链条，因此，在`test_tipc/output`文件夹有以下文件：

```
test_tipc/output/
|- results_python.log    # 运行指令状态的日志
|- norm_train_gpus_0_autocast_null/  # GPU 0号卡上正常训练的训练日志和模型保存文件夹
|- pact_train_gpus_0_autocast_null/  # GPU 0号卡上量化训练的训练日志和模型保存文件夹
......
|- python_infer_cpu_usemkldnn_True_threads_1_batchsize_1.log  # CPU上开启Mkldnn线程数设置为1，测试batch_size=1条件下的预测运行日志
|- python_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log # GPU上开启TensorRT，测试batch_size=1的半精度预测日志
......
```

其中`results_python.log`中包含了每条指令的运行状态，如果运行成功会输出：
```
Run successfully with command - python3.7 main.py --model=lvvit_t --drop_path=0.1 --lr=1.6e-3 --warmup_lr=1e-6 --min_lr=1e-5 --t_in_epochs --mixup=0 --cutmix=0 --crop_pct=0.9 --train_interpolation=random --data_path=./dataset/ILSVRC2012/ --cls_label_path_train=./dataset/ILSVRC2012/train_list.txt --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --token_label --token_label_data=./dataset/ILSVRC2012/label_top5_train_nfnet/ --token_label_size=14 --dist_eval --output_dir=./test_tipc/output/norm_train_gpus_0_autocast_null/lvvit_t --epochs=2 --batch_size=8 !
Run successfully with command - python3.7 eval.py --model=lvvit_t --crop_pct=0.9 --data_path=./dataset/ILSVRC2012/ --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/lvvit_t/checkpoint-latest.pd !
Run successfully with command - python3.7 export_model.py --model=lvvit_t --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/lvvit_t/checkpoint-latest.pd --output=./test_tipc/output/norm_train_gpus_0_autocast_null !
Run successfully with command - python3.7 infer.py --use_gpu=True --use_tensorrt=False --precision=fp32 --model_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdmodel --batch_size=1 --input_file=./dataset/ILSVRC2012/val  --params_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdiparams > ./test_tipc/output/python_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log 2>&1 !
......
```
如果运行失败，会输出：
```
Run failed with command - python eval.py --model=lvvit_t --data_path=./dataset/ILSVRC2012/ --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --resume=./test_tipc/output/norm_train_gpus_0,1_autocast_null/lvvit_t/checkpoint-latest.pd    !
Run failed with command - python export_model.py --model=lvvit_t --resume=./test_tipc/output/norm_train_gpus_0,1_autocast_null/lvvit_t/checkpoint-latest.pd --output=./test_tipc/output/norm_train_gpus_0,1_autocast_null!  
......
```
可以很方便的根据`results_python.log`中的内容判定哪一个指令运行错误。
