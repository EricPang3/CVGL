import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # 将模型设为训练模式
    model.train()
    
    losses = AverageMeter()
    
    # 启动进度条前稍作等待
    time.sleep(0.1)
    
    # 在第一步之前清空梯度
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # 遍历一个训练轮次中的所有批次
    for query, reference, ids in bar:
        
        if scaler:
            with autocast():
            
                # 将批次数据移至指定设备   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # 前向传播
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # 梯度裁剪 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # 更新模型参数（权重）
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度，为下一步做准备
            optimizer.zero_grad()
            
            # 学习率调度器
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # 将批次数据移至指定设备   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # 前向传播
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # 通过反向传播计算梯度
            loss.backward()
            
            # 梯度裁剪 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # 更新模型参数（权重）
            optimizer.step()
            # 清空梯度，为下一步做准备
            optimizer.zero_grad()
            
            # 学习率调度器
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # 启动进度条前稍作等待
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # 使用 FP32 计算归一化
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # 将特征保存为 FP32，用于相似度计算
            img_features_list.append(img_feature.to(torch.float32))
      
        # 将特征保留在 GPU 上
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list