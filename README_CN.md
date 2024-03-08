## ESM-GearNet

蛋白序列（ESM）+ 结构（AlphaFold）

## 安装

1. 创建环境

```shell
conda create -n esm+ python=3.8
```

2. 安装torch

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

3. 安装torch-scatter

```shell
conda install pytorch-scatter -c pyg
```

4. 安装torchdrug

```shell
conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg

```

4. 安装pyg
```shell
conda install pyg -c pyg
```

1. 安装其他必要包

```shell
conda install easydict pyyaml -c conda-forge
conda install transformers==4.14.1 tokenizers==0.10.3 -c huggingface 
pip install atom3d
```

## 使用

默认情况下，我们将使用 AlphaFold Datase 进行预训练。要使用多视图对比度预训练 ESM-GearNet，请使用以下命令。类似地，所有数据集将在您第一次运行代码时自动下载到代码中并进行预处理。

```shell
# Run pre-training
python -m torch.distributed.launch --nproc_per_node=2 script/pretrain.py -c config/pretrain/mc_esm_gearnet.yaml
```

您可以通过 --ckpt 参数从保存的检查点加载模型权重，然后在下游任务上微调模型。请记住首先取消注释配置文件中的`model_checkpoint: {{ ckpt }}`行。

```shell
python -m torch.distributed.launch --nproc_per_node=3 script/downstream.py -c config/EC/esm_gearnet.yaml --ckpt <path_to_your_model>
```
