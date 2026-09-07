import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random


# -----------------------------------------------------------------------------#
# SUES-200 dataset (Zhu et al., TCSVT 2023 - SUES-200-Benchmark)              #
#                                                                              #
# Folder names inside the official release (top folder name is arbitrary,      #
# e.g. <data_folder> = ./data/SUES200 as used by train_sues200.py)             #
#                                                                              #
#   <data_folder>/                                                             #
#   ├── satellite-view/<place>/0.png            one satellite image per place  #
#   └── drone_view_512/<place>/<altitude>/*.jpg 50 drone images per place,     #
#                                                altitude in {150,200,250,300} #
#                                                                              #
# The official benchmark trains/evaluates one model per altitude and uses a    #
# fixed 120/80 (60/40) train/test split over the 200 places (see the official #
# indexs.yaml). The classes below read the raw tree directly - no additional   #
# Training/Testing folder materialization is required.                         #
# -----------------------------------------------------------------------------#

SATELLITE_VIEW = "satellite-view"
DRONE_VIEW = "drone_view_512"
ALTITUDES = ["150", "200", "250", "300"]

# Official 120 training places of the SUES-200 benchmark (SUES-200-Benchmark,
# Reza-Zhu et al., TCSVT 2023); the remaining 80 places are used for testing.
OFFICIAL_TRAIN_PLACES = [
    '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0009',
    '0010', '0011', '0013', '0015', '0016', '0020', '0021', '0024',
    '0025', '0026', '0027', '0028', '0029', '0030', '0031', '0032',
    '0033', '0034', '0036', '0037', '0039', '0043', '0044', '0045',
    '0051', '0055', '0056', '0057', '0058', '0060', '0064', '0066',
    '0067', '0069', '0070', '0073', '0074', '0076', '0077', '0080',
    '0081', '0082', '0084', '0085', '0086', '0087', '0088', '0089',
    '0090', '0091', '0094', '0096', '0097', '0100', '0101', '0102',
    '0103', '0104', '0105', '0106', '0113', '0116', '0118', '0119',
    '0120', '0123', '0125', '0126', '0127', '0129', '0132', '0134',
    '0137', '0139', '0141', '0142', '0145', '0146', '0147', '0148',
    '0149', '0151', '0152', '0155', '0156', '0157', '0158', '0159',
    '0160', '0163', '0166', '0167', '0168', '0169', '0171', '0174',
    '0179', '0180', '0182', '0183', '0184', '0185', '0186', '0187',
    '0191', '0192', '0193', '0196', '0197', '0198', '0199', '0200',
]


def get_places(data_folder):
    """All place ids ('0001'...'0200') present in the satellite view folder."""
    places = []
    sat_root = os.path.join(data_folder, SATELLITE_VIEW)
    for name in sorted(os.listdir(sat_root)):
        if os.path.isdir(os.path.join(sat_root, name)):
            places.append(name)
    return places


def get_sues200_split(data_folder):
    """Return (train_places, test_places) following the official 120/80 split."""
    train_places = [p for p in OFFICIAL_TRAIN_PLACES if os.path.isdir(
        os.path.join(data_folder, SATELLITE_VIEW, p))]
    test_places = [p for p in get_places(data_folder) if p not in train_places]
    return train_places, test_places


class SUES200DatasetTrain(Dataset):

    def __init__(self,
                 data_folder,
                 altitude=150,
                 train_places=None,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.altitude = str(altitude)
        if self.altitude not in ALTITUDES:
            raise ValueError("altitude must be one of {} but is {}".format(ALTITUDES, self.altitude))

        sat_root = os.path.join(data_folder, SATELLITE_VIEW)
        drone_root = os.path.join(data_folder, DRONE_VIEW)

        # use only folders that exist for both satellite and drone view
        self.ids = [p for p in get_places(data_folder)
                    if (train_places is None or p in train_places)
                    and os.path.isdir(os.path.join(drone_root, p, self.altitude))]
        self.ids.sort()

        self.pairs = []

        for idx in self.ids:

            sat_files = sorted(os.listdir(os.path.join(sat_root, idx)))
            if len(sat_files) == 0:
                continue
            query_img = os.path.join(sat_root, idx, sat_files[0])

            drone_dir = os.path.join(drone_root, idx, self.altitude)
            drone_files = sorted(f for f in os.listdir(drone_dir)
                                 if os.path.isfile(os.path.join(drone_dir, f)))

            for g in drone_files:
                self.pairs.append((idx, query_img, os.path.join(drone_dir, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one satellite image per place
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self,):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:

                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class SUES200DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 view,
                 altitude=150,
                 places=None,
                 transforms=None,
                 given_sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.altitude = str(altitude)
        if self.altitude not in ALTITUDES:
            raise ValueError("altitude must be one of {} but is {}".format(ALTITUDES, self.altitude))

        if view not in ['drone', 'satellite']:
            raise ValueError("view must be 'drone' or 'satellite' but is {}".format(view))

        self.view = view
        self.transforms = transforms
        self.given_sample_ids = given_sample_ids
        self.gallery_n = gallery_n

        self.images = []
        self.sample_ids = []

        if view == 'drone':
            root = os.path.join(data_folder, DRONE_VIEW)
        else:
            root = os.path.join(data_folder, SATELLITE_VIEW)

        self.ids = [p for p in get_places(data_folder)
                    if places is None or p in places]

        if self.gallery_n > 0:
            self.ids = self.ids[:self.gallery_n]

        for sample_id in self.ids:

            if view == 'drone':
                img_dir = os.path.join(root, sample_id, self.altitude)
            else:
                img_dir = os.path.join(root, sample_id)

            if not os.path.isdir(img_dir):
                continue

            files = sorted(f for f in os.listdir(img_dir)
                           if os.path.isfile(os.path.join(img_dir, f)))

            for file in files:
                self.images.append(os.path.join(img_dir, file))
                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*img_size[0]),
                                                               max_width=int(0.2*img_size[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*img_size[0]),
                                                               min_width=int(0.1*img_size[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.4, p=1.0),
                                                 A.CoarseDropout(max_holes=25,
                                                                 max_height=int(0.2*img_size[0]),
                                                                 max_width=int(0.2*img_size[0]),
                                                                 min_holes=10,
                                                                 min_height=int(0.1*img_size[0]),
                                                                 min_width=int(0.1*img_size[0]),
                                                                 p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms
