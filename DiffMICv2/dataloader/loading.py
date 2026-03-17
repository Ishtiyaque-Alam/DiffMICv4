import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_erosion
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import filters
import numpy as np
import imageio
import dataloader.transforms as trans
import json, numbers
from glob import glob
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class BUDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                # trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                # trans.adjust_light(),
                transforms.ToTensor(),
                # lambda x: x*255
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # lambda x: x*255
                ])


    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size


class APTOSDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                # transforms.GaussianBlur(3),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label

    def __len__(self):
        return self.size



class ISICDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)

        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                #trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size


class ChestXrayDataSet(Dataset):
    def __init__(self, image_list_file, train=True):
        """
        self:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        data_dir = "dataset/chest/all/images/images"
        # data_dir = "/home/yijun/project/DiffMIC/dataset/chest/all/images_enhanced"
        self.trainsize = (256, 256)
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                label.append(1) if (np.array(label)==0).all() else label.append(0)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        #print(len(self.image_names))
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        if train:
            self.transform_center = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        
                                        # transforms.RandomCrop(224),
                                        # transforms.GaussianBlur(3),
                                        trans.RandomHorizontalFlip(),
                                        trans.RandomRotation(20),
                                        transforms.ToTensor(),
                                        normalize
                                        #transforms.TenCrop(224),
                                        #transforms.Lambda
                                        #(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda
                                        #(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])
        else:
            # self.image_names = image_names[:1000]
            # self.labels = labels[:1000]
            self.transform_center = transforms.Compose([
                                        transforms.Resize(224),
                                        # transforms.TenCrop(224),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.RandomHorizontalFlip()(crop) for crop in crops])),
                                        # # transforms.Lambda
                                        # # (lambda crops: torch.stack([transforms.RandomRotation(20)(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])

    def __getitem__(self, index):
        """
        self:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        image = self.transform_center(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


class HAM10000Dataset(Dataset):
    """
    HAM10000 dataset loader.
    Reads GroundTruth.csv (one-hot columns for 7 lesion types) and images directory.
 
    Balancing strategy: WeightedRandomSampler
        - Computes per-class sampling weights so every class is seen equally per batch
        - Does NOT duplicate or delete images
        - Minority class images seen with different augmentation each time (free diversity)
        - Use get_sampler() to get the sampler for your DataLoader
 
    GroundTruth.csv format:
        image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
        ISIC_0024306,0,1,0,0,0,0,0
        ...
 
    Usage:
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, csv_path)
    """
 
    CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
 
    def __init__(
        self,
        data_dir,
        csv_path,
        split       = 'train',  # 'train', 'val', or 'test'
        train_ratio = 0.70,
        val_ratio   = 0.15,
        seed        = 2000
    ):
        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        assert train_ratio + val_ratio < 1.0, \
            "train_ratio + val_ratio must be < 1.0"
 
        self.split     = split
        self.trainsize = (224, 224)
 
        # ── Load CSV ──────────────────────────────────────────────────────────
        df        = pd.read_csv(csv_path)
        labels    = df[self.CLASS_NAMES].values.argmax(axis=1)
        image_ids = df['image'].values
 
        # ── Build (image_path, label) pairs ───────────────────────────────────
        all_items = []
        for img_id, label in zip(image_ids, labels):
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                candidate = os.path.join(data_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                candidate = os.path.join(data_dir, img_id)
                img_path  = candidate if os.path.exists(candidate) \
                            else os.path.join(data_dir, img_id + '.jpg')
            all_items.append((img_path, int(label)))
 
        all_paths  = [x[0] for x in all_items]
        all_labels = [x[1] for x in all_items]
 
        # ── Stratified 70 / 15 / 15 split ────────────────────────────────────
        test_ratio = 1.0 - train_ratio - val_ratio
 
        # Step 1: carve out test (15%)
        train_val_paths,  test_paths,  \
        train_val_labels, test_labels = train_test_split(
            all_paths, all_labels,
            test_size    = test_ratio,
            stratify     = all_labels,
            random_state = seed
        )
 
        # Step 2: split remainder into train (70%) and val (15%)
        val_fraction = val_ratio / (train_ratio + val_ratio)
        train_paths,  val_paths,  \
        train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size    = val_fraction,
            stratify     = train_val_labels,
            random_state = seed
        )
 
        # ── Select the right split ────────────────────────────────────────────
        split_map = {
            'train': (train_paths, train_labels),
            'val':   (val_paths,   val_labels),
            'test':  (test_paths,  test_labels),
        }
        paths, labels = split_map[split]
 
        self.data_list = list(zip(paths, labels))
        self.labels    = np.array(labels)   # kept separately for get_sampler()
        self.size      = len(self.data_list)
 
        # ── Print split summary ───────────────────────────────────────────────
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\n[HAM10000Dataset] split='{split}' | {self.size} total images")
        print(f"  Class distribution:")
        for cls_idx, cnt in zip(unique, counts):
            print(f"    {self.CLASS_NAMES[cls_idx]:6s}: {cnt:5d} "
                  f"({'%.1f' % (cnt/self.size*100)}%)")
 
        # ── Transforms ───────────────────────────────────────────────────────
        normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        )
 
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.trainsize),
                # Geometric — full rotation valid for dermoscopy
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                # Color — device/lighting/skin tone variation
                transforms.ColorJitter(
                    brightness = 0.3,
                    contrast   = 0.3,
                    saturation = 0.3,
                    hue        = 0.1
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
                # Cutout — forces model not to rely on any single region
                transforms.RandomErasing(
                    p     = 0.3,
                    scale = (0.02, 0.1),
                    ratio = (0.3, 3.3)
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                normalize,
            ])
 
    # ── WeightedRandomSampler ─────────────────────────────────────────────────
    def get_sampler(self):
        """
        Returns a WeightedRandomSampler for balanced training.
        Only call this for the training split.
 
        How it works:
            weight per class  = 1 / class_count
            weight per sample = weight of its class
 
            Raw HAM10000 example:
                NV:  6705 images → weight = 0.000149  (sampled rarely)
                DF:   115 images → weight = 0.008696  (sampled often)
 
            Each class appears roughly equally in every batch after sampling.
 
        IMPORTANT: Do NOT use shuffle=True in DataLoader when using this sampler.
        """
        assert self.split == 'train', \
            "get_sampler() should only be used for the training split"
 
        class_counts  = np.bincount(self.labels)
        class_weights = 1.0 / class_counts.astype(np.float32)
        sample_weights = class_weights[self.labels]
 
        print(f"\n[WeightedRandomSampler] Per-class weights:")
        for i, cls in enumerate(self.CLASS_NAMES):
            print(f"  {cls:6s}: count={class_counts[i]:5d}  "
                  f"weight={class_weights[i]:.6f}")
 
        return WeightedRandomSampler(
            weights     = torch.from_numpy(sample_weights),
            num_samples = len(sample_weights),
            replacement = True   # must be True to oversample minority classes
        )
 
    def __getitem__(self, index):
        img_path, label = self.data_list[index]
        img             = Image.open(img_path).convert('RGB')
        return self.transform(img), label
 
    def __len__(self):
        return self.size
 
 
# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloaders(data_dir, csv_path, batch_size=32, num_workers=4, seed=2000):
    """
    Creates all three DataLoaders with balancing applied to train only.
 
    Args:
        data_dir:    Path to image directory
        csv_path:    Path to GroundTruth.csv
        batch_size:  Batch size (32 is optimal for T4 + EfficientSAM)
        num_workers: Parallel workers (4 is safe for Colab)
        seed:        Reproducibility seed
 
    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = HAM10000Dataset(data_dir, csv_path, split='train', seed=seed)
    val_ds   = HAM10000Dataset(data_dir, csv_path, split='val',   seed=seed)
    test_ds  = HAM10000Dataset(data_dir, csv_path, split='test',  seed=seed)
 
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = train_ds.get_sampler(),  # balanced — replaces shuffle=True
        num_workers = num_workers,
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True
    )
 
    return train_loader, val_loader, test_loader
