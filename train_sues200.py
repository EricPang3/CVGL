import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:

    # 模型
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'

    # 模型的输入图像尺寸
    img_size: int = 384

    # 训练
    mixed_precision: bool = True         # 混合精度
    custom_sampling: bool = True         # 自定义采样，保证同一批次地点编号不重复
    seed = 1
    epochs: int = 1
    batch_size: int = 24                # 注意实际批量大小 real_batch_size = 2 * batch_size
    verbose: bool = True                 # 开启详细输出，False 时会关闭显示
    gpu_ids: tuple = (0,)           # 用于训练的 GPU 编号

    # 评估
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1          # 每隔 n 个训练轮次进行一次评估
    normalize_features: bool = True      # 是否对特征进行归一化
    eval_gallery_n: int = -1             # -1 表示使用全部图像，也可指定整数数量

    # 优化器
    clip_grad = 100.                     # 梯度裁剪阈值，不裁剪时设为 None，否则设为浮点数阈值
    decay_exclue_bias: bool = False      # 是否在权重衰减时排除偏置参数
    grad_checkpointing: bool = False     # 梯度检查点：是否用额外计算换更少显存占用

    # 损失函数
    label_smoothing: float = 0.1         # 标签平滑：将部分概率分配给其他类别，避免过拟合和提高模型的泛化能力。设置为 0 表示不使用标签平滑。

    # 学习率
    lr: float = 0.001                    # ViT 使用 1 * 10^-4，CNN 使用 1 * 10^-1
    scheduler: str = "cosine"            # 可选多项式衰减（"polynomial"）、余弦衰减（"cosine"）、恒定学习率（"constant"）或 None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  仅用于多项式衰减（"polynomial"）

    # 数据集
    dataset: str = 'SUES200-D2S'        # 任务方向，可选 'SUES200-D2S'（无人机到卫星）或 'SUES200-S2D'（卫星到无人机）
    data_folder: str = "./data/SUES-200" # 数据集路径
    altitude: int = 150                 # 图像高度，可选 150、200、250、300

    # 图像增强
    prob_flip: float = 0.5              # 同时水平翻转卫星图像和无人机图像的概率

    # 模型权重的保存路径
    model_path: str = "./checkpoints/sues200"

    # 训练前进行评估
    zero_shot: bool = False

    # 初始权重路径
    checkpoint_start = None

    # 在 Windows 系统上将 num_workers 设为 0
    num_workers: int = 0 if os.name == 'nt' else 4

    # 如果 GPU 可用，则使用 GPU
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 提高运行性能
    cudnn_benchmark: bool = True

    # 使 cuDNN 使用确定性算法
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# 训练配置                                                                #
#-----------------------------------------------------------------------------#

config = Configuration()

if config.dataset == 'SUES200-D2S':
    config.query_folder_train = os.path.join(config.data_folder, 'Training', str(config.altitude), 'satellite')
    config.gallery_folder_train = os.path.join(config.data_folder, 'Training', str(config.altitude), 'drone')
    config.query_folder_test = os.path.join(config.data_folder, 'Testing', str(config.altitude), 'query_drone')
    config.gallery_folder_test = os.path.join(config.data_folder, 'Testing', str(config.altitude), 'gallery_satellite')
elif config.dataset == 'SUES200-S2D':
    config.query_folder_train = os.path.join(config.data_folder, 'Training', str(config.altitude), 'satellite')
    config.gallery_folder_train = os.path.join(config.data_folder, 'Training', str(config.altitude), 'drone')
    config.query_folder_test = os.path.join(config.data_folder, 'Testing', str(config.altitude), 'query_satellite')
    config.gallery_folder_test = os.path.join(config.data_folder, 'Testing', str(config.altitude), 'gallery_drone')


if __name__ == '__main__':

    model_path = "{}/{}/{}/{}/{}".format(config.model_path,
                                       config.altitude,
                                       config.dataset,
                                       config.model,
                                       time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(Path(__file__).resolve(), "{}/train.py".format(model_path))

    # 将打印输出同时写入控制台和日志文件
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # 模型                                                                       #
    #-----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    # 启用梯度检查点以节省显存
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # 加载已有的模型权重
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # 数据并行多GPU训练
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # 将模型移至指定设备
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))


    #-----------------------------------------------------------------------------#
    # 数据加载器                                                                  #
    #-----------------------------------------------------------------------------#

    # 图像预处理和增强
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # 训练
    train_dataset = U1652DatasetTrain(query_folder=config.query_folder_train,
                                      gallery_folder=config.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # 卫星参考图像
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # 地面查询图像测试
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    #-----------------------------------------------------------------------------#
    # 损失函数                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # 优化器                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # 学习率调度器                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))


    #-----------------------------------------------------------------------------#
    # 零样本评估                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    #-----------------------------------------------------------------------------#
    # 打乱数据顺序                                                                     #
    #-----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    #-----------------------------------------------------------------------------#
    # 训练                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0


    for epoch in range(1, config.epochs+1):

        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))


        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # 评估
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))


        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
