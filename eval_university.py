import os
import argparse
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.evaluate.university import evaluate
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='在 University-1652 上评估',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # 模型
        parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='模型名称')

        # 覆盖模型的输入图像尺寸
        parser.add_argument('--img_size', default=384, type=int, help='输入图像尺寸')

        # 评估
        parser.add_argument('--batch_size', default=128, type=int, help='评估批量大小')
        parser.add_argument('--verbose', default=True, action=argparse.BooleanOptionalAction, help='输出详细信息')
        parser.add_argument('--gpu_ids', default=(0,), type=int, nargs='+', help='GPU 编号列表，例如 --gpu_ids 0 1；单卡设备可用 --device cuda:1 指定')
        parser.add_argument('--normalize_features', default=True, action=argparse.BooleanOptionalAction, help='评估时对特征归一化')
        parser.add_argument('--eval_gallery_n', default=-1, type=int, help='gallery 图像数量；-1 使用全部')

        # 数据集
        parser.add_argument('--dataset', default='U1652-D2S', type=str, choices=('U1652-D2S', 'U1652-S2D'), help='检索任务方向')
        parser.add_argument('--data_folder', default="./data/U1652", type=str, help='数据集根目录，相对路径从当前工作目录计算')

        # 用于初始化模型的检查点
        parser.add_argument('--checkpoint_start', default='pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth', type=str, help='要评估的模型权重路径')

        # 在 Windows 系统上将 num_workers 设为 0
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int, help='数据加载进程数')

        # 如果 GPU 可用，则使用 GPU
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备，例如 cuda、cuda:1 或 cpu')

        parser.parse_args(namespace=self)
        self.gpu_ids = tuple(self.gpu_ids)
    

#-----------------------------------------------------------------------------#
# 配置                                                                      #
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
    

    # 加载预训练检查点    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # 数据并行
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

    # 图像变换
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    
    # 卫星参考图像
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
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
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
   

    print("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
 
