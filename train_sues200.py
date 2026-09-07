import os
import time
import shutil
import sys
import gc
import torch
from dataclasses import dataclass, field
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.sues200 import SUES200DatasetTrain, SUES200DatasetEval, get_transforms, get_places, get_sues200_split, ALTITUDES
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.sues200 import evaluate
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


# Keep references to all altitude Loggers: Logger.__del__ closes the console stream,
# so a garbage-collected Logger would break the console for the next altitude.
loggers = []


@dataclass
class Configuration:

    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'

    # Override model image size
    img_size: int = 384

    # Training
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    # Epochs 说明：默认 1 与 train_university.py 相同（仓库用它做快速验证/调试）。
    # SUES-200 单高度每 epoch 只有约 187 步（120 地点 x 50 图 / batch 32），1 epoch 远远不够：
    #   - 仓库其它数据集默认：CVUSA/CVACT/VIGOR = 40
    #   - 官方 SUES-200-Benchmark：60（MultiStepLR [20,40]）
    #   - 建议：先用 epochs=1 验证管线，正式训练设 40~80（例如 40）
    epochs: int = 1
    batch_size: int = 32                 # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)                # GPU ids for training

    # Eval
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = True      # Gradient Checkpointing

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               # only for "polynomial"

    # Dataset
    dataset: str = 'SUES200-D2S'         # 'SUES200-D2S' (drone -> satellite) | 'SUES200-S2D' (satellite -> drone)
    altitudes: list = field(default_factory=lambda: [150, 200, 250, 300])  # train & evaluate one model per
                                           # altitude (official protocol); results are saved separately per altitude
    data_folder: str = "./data/SUES200"          # same convention as the other datasets in this repo
    split_file: str = None               # None -> official 120/80 split | path to txt (one place id per line)
    eval_gallery_mode: str = 'all'       # 'all' -> official protocol (gallery = 200 places) | 'test' -> gallery = 80 test places

    # Augment Images
    prob_flip: float = 0.5               # flipping the sat image and drone image simultaneously

    # Savepath for model checkpoints
    model_path: str = "./sues200"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 8

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration()

if config.dataset == 'SUES200-D2S':
    # evaluation: drone images (test places) as query, satellite images as gallery
    config.query_view = 'drone'
    config.gallery_view = 'satellite'
elif config.dataset == 'SUES200-S2D':
    # evaluation: satellite images (test places) as query, drone images as gallery
    config.query_view = 'satellite'
    config.gallery_view = 'drone'


def get_split():
    """Official 120/80 place split, or a custom one if config.split_file is given."""

    if config.split_file is None:
        train_places, test_places = get_sues200_split(config.data_folder)
    else:
        with open(config.split_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        available = set(get_places(config.data_folder))
        train_places = [p for p in lines if p in available]
        test_places = [p for p in get_places(config.data_folder) if p not in train_places]
    return train_places, test_places


def run_altitude(altitude, train_places, test_places):

    altitude = str(altitude)
    if altitude not in ALTITUDES:
        raise ValueError("altitude must be one of {} but is {}".format(ALTITUDES, altitude))

    model_path = "{}/{}/{}m_{}".format(config.model_path,
                                       config.model,
                                       altitude,
                                       time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file of this altitude
    logger = Logger(os.path.join(model_path, 'log.txt'))
    sys.stdout = logger
    loggers.append(logger)

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    print("\n{}[Altitude: {}m]{}".format(30*"-", altitude, 30*"-"))

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
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

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # Train
    train_dataset = SUES200DatasetTrain(data_folder=config.data_folder,
                                        altitude=altitude,
                                        train_places=train_places,
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

    # Query Images Test
    query_dataset_test = SUES200DatasetEval(data_folder=config.data_folder,
                                            view=config.query_view,
                                            altitude=altitude,
                                            places=test_places,
                                            transforms=val_transforms,
                                            )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Gallery Images Test
    if config.eval_gallery_mode == 'test':
        gallery_places = test_places
    else:
        gallery_places = None            # official protocol: all 200 places as gallery

    gallery_dataset_test = SUES200DatasetEval(data_folder=config.data_folder,
                                              view=config.gallery_view,
                                              altitude=altitude,
                                              places=gallery_places,
                                              transforms=val_transforms,
                                              gallery_n=config.eval_gallery_n,
                                              )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    print("\nAltitude: {}m - Train Places: {} - Test Places: {}".format(altitude, len(train_places), len(test_places)))
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
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
    # optimizer                                                                   #
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
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
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
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0
    best_metrics = None

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

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test, metrics = evaluate(config=config,
                                        model=model,
                                        query_loader=query_dataloader_test,
                                        gallery_loader=gallery_dataloader_test,
                                        ranks=[1, 5, 10],
                                        step_size=1000,
                                        cleanup=True,
                                        return_metrics=True)

            if r1_test > best_score:

                best_score = r1_test
                best_metrics = metrics

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

    if best_metrics is not None:
        print("\nBest {}m: R@1: {:.4f} - AP: {:.4f}".format(altitude, best_score*100, best_metrics['AP']))
    else:
        print("\nBest {}m: no improving epoch found (R@1 = 0)".format(altitude))

    # free memory before the next altitude
    del model, train_dataset, query_dataset_test, gallery_dataset_test
    gc.collect()
    torch.cuda.empty_cache()

    if best_metrics is not None:
        return float(best_score), float(best_metrics['AP'])
    return 0.0, 0.0


if __name__ == '__main__':

    orig_stdout = sys.stdout

    train_places, test_places = get_split()

    print("Dataset: {} - Altitudes: {}".format(config.dataset, config.altitudes))
    print("Data Folder: {}".format(config.data_folder))
    print("Train Places: {} - Test Places: {}".format(len(train_places), len(test_places)))

    results = []

    for altitude in config.altitudes:

        # make sure the console stream is not captured by the previous altitude Logger
        sys.stdout = orig_stdout

        r1, ap = run_altitude(altitude, train_places, test_places)
        results.append((altitude, r1, ap))

    #-----------------------------------------------------------------------------#
    # Summary                                                                     #
    #-----------------------------------------------------------------------------#
    sys.stdout = orig_stdout

    if not os.path.exists("{}/{}".format(config.model_path, config.model)):
        os.makedirs("{}/{}".format(config.model_path, config.model))

    summary_path = "{}/{}/results_{}.txt".format(config.model_path,
                                                 config.model,
                                                 time.strftime("%Y%m%d_%H%M%S"))

    print("\n{}[Results]{}".format(30*"-", 30*"-"))
    print("Summary saved to: {}".format(summary_path))

    with open(summary_path, 'w') as f:
        f.write("Dataset: {}\n".format(config.dataset))
        f.write("Model: {} - Image Size: {}\n".format(config.model, config.img_size))
        f.write("Epochs: {} - Batch Size: {} - Altitudes: {}\n".format(config.epochs, config.batch_size, config.altitudes))
        f.write("Metric selection: best Recall@1 on the test set\n\n")
        f.write("Altitude  Recall@1   AP\n")
        for altitude, r1, ap in results:
            f.write("{:<8}  {:<8.4f}  {:.4f}\n".format(altitude, r1*100, ap))
            print("Altitude {}m - Best Recall@1: {:.4f} - AP: {:.4f}".format(altitude, r1*100, ap))
