import os
import argparse
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


def optional_float(value):
    """将命令行浮点数或 None 转换为配置值。"""
    return None if value.lower() == "none" else float(value)


@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='在 University-1652 上训练并评估',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # 模型
        parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='模型名称')

        # 模型的输入图像尺寸
        parser.add_argument('--img_size', default=384, type=int, help='输入图像尺寸')

        # 训练
        parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction, help='混合精度训练')
        parser.add_argument('--custom_sampling', default=True, action=argparse.BooleanOptionalAction, help='每个 batch 内地点不重复的自定义采样')
        parser.add_argument('--seed', default=1, type=int, help='随机种子')
        parser.add_argument('--epochs', default=1, type=int, help='训练轮数')
        parser.add_argument('--batch_size', default=24, type=int, help='批量大小；训练时为图像对数量')
        parser.add_argument('--verbose', default=True, action=argparse.BooleanOptionalAction, help='输出详细信息')
        parser.add_argument('--gpu_ids', default=(0,), type=int, nargs='+', help='GPU 编号列表，例如 --gpu_ids 0 1；单卡设备可用 --device cuda:1 指定')

        # 评估
        parser.add_argument('--batch_size_eval', default=128, type=int, help='评估批量大小')
        parser.add_argument('--eval_every_n_epoch', default=1, type=int, help='每隔多少轮评估一次')
        parser.add_argument('--normalize_features', default=True, action=argparse.BooleanOptionalAction, help='评估时对特征归一化')
        parser.add_argument('--eval_gallery_n', default=-1, type=int, help='gallery 图像数量；-1 使用全部')

        # 优化器
        parser.add_argument('--clip_grad', default=100., type=optional_float, help='梯度裁剪阈值；None 或 0 关闭')
        parser.add_argument('--decay_exclue_bias', default=False, action=argparse.BooleanOptionalAction, help='在权重衰减中排除偏置参数')
        parser.add_argument('--grad_checkpointing', default=False, action=argparse.BooleanOptionalAction, help='启用梯度检查点')

        # 损失函数
        parser.add_argument('--label_smoothing', default=0.1, type=float, help='标签平滑系数')

        # 学习率
        parser.add_argument('--lr', default=0.001, type=float, help='学习率')
        parser.add_argument('--scheduler', default="cosine", type=str.lower, choices=('polynomial', 'cosine', 'constant', 'none'), help='学习率调度器；none 关闭')
        parser.add_argument('--warmup_epochs', default=0.1, type=float, help='warmup 轮数，可为小数')
        parser.add_argument('--lr_end', default=0.0001, type=float, help='多项式调度器的最终学习率')

        # 数据集
        parser.add_argument('--dataset', default='U1652-D2S', type=str, choices=('U1652-D2S', 'U1652-S2D'), help='检索任务方向')
        parser.add_argument('--data_folder', default="./data/U1652", type=str, help='数据集根目录，相对路径从当前工作目录计算')

        # 图像增强
        parser.add_argument('--prob_flip', default=0.5, type=float, help='配对图像同时水平翻转的概率')

        # 模型权重的保存路径
        parser.add_argument('--model_path', default="./university", type=str, help='训练结果保存根目录')

        # 训练前进行评估
        parser.add_argument('--zero_shot', default=False, action=argparse.BooleanOptionalAction, help='训练前先评估')

        # 初始权重路径
        parser.add_argument('--checkpoint_start', default=None, type=str, help='初始模型权重路径；训练时不恢复优化器')

        # 在 Windows 系统上将 num_workers 设为 0
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int, help='数据加载进程数')

        # 如果 GPU 可用，则使用 GPU
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备，例如 cuda、cuda:1 或 cpu')

        # 提高运行性能
        parser.add_argument('--cudnn_benchmark', default=True, action=argparse.BooleanOptionalAction, help='启用 cuDNN benchmark')

        # 使 cuDNN 使用确定性算法
        parser.add_argument('--cudnn_deterministic', default=False, action=argparse.BooleanOptionalAction, help='使用 cuDNN 确定性算法')

        parser.parse_args(namespace=self)
        self.gpu_ids = tuple(self.gpu_ids)
        if self.scheduler == "none":
            self.scheduler = None


#-----------------------------------------------------------------------------#
# 训练配置                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'U1652-D2S':
    config.query_folder_train = os.path.join(config.data_folder, 'train', 'satellite')
    config.gallery_folder_train = os.path.join(config.data_folder, 'train', 'drone')
    config.query_folder_test = os.path.join(config.data_folder, 'test', 'query_drone')
    config.gallery_folder_test = os.path.join(config.data_folder, 'test', 'gallery_satellite')
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = os.path.join(config.data_folder, 'train', 'satellite')
    config.gallery_folder_train = os.path.join(config.data_folder, 'train', 'drone')
    config.query_folder_test = os.path.join(config.data_folder, 'test', 'query_satellite')
    config.gallery_folder_test = os.path.join(config.data_folder, 'test', 'gallery_drone')


if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
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
