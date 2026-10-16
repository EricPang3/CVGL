import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.evaluate.university import evaluate
from sample4geo.model import TimmModel


@dataclass
class Configuration:

    # 模型
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # 覆盖模型的输入图像尺寸
    img_size: int = 384
    
    # 评估
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 表示使用全部图像，也可指定整数数量
    
    # 数据集
    dataset: str = 'U1652-D2S'           # 无人机到卫星检索（'U1652-D2S'）或卫星到无人机检索（'U1652-S2D'）
    data_folder: str = "./data/U1652"
    
    # 用于初始化模型的检查点
    checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
  
    # 在 Windows 系统上将 num_workers 设为 0
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # 如果 GPU 可用，则使用 GPU
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

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
 
