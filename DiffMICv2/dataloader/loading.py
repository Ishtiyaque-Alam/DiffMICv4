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
    Supports 80/20 stratified train-test split.
    
    GroundTruth.csv format:
        image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
        ISIC_0024306,0,1,0,0,0,0,0
        ...
    """
    # Class ordering matches GroundTruth.csv column order
    CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __init__(self, data_dir, csv_path, train=True, split_ratio=0.8, seed=2000):
        """
        Args:
            data_dir:    Path to directory containing images (e.g. .jpg files)
            csv_path:    Path to GroundTruth.csv
            train:       If True, use training split; else use test split
            split_ratio: Fraction of data for training (default 0.8)
            seed:        Random seed for reproducible split
        """
        import pandas as pd
        self.trainsize = (224, 224)
        self.train = train

        # Read CSV
        df = pd.read_csv(csv_path)
        # Get labels from one-hot columns
        label_cols = self.CLASS_NAMES
        # argmax across one-hot columns to get class index
        labels = df[label_cols].values.argmax(axis=1)
        image_ids = df['image'].values

        # Build (image_path, label) pairs
        all_items = []
        for img_id, label in zip(image_ids, labels):
            # Try common extensions
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                candidate = os.path.join(data_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                # Try without extension (file might already have extension in name)
                candidate = os.path.join(data_dir, img_id)
                if os.path.exists(candidate):
                    img_path = candidate
                else:
                    # Fallback: just use jpg
                    img_path = os.path.join(data_dir, img_id + '.jpg')
            all_items.append((img_path, int(label)))

        # 80/20 stratified split
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_items))
        rng.shuffle(indices)
        split_idx = int(len(indices) * split_ratio)

        if train:
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        self.data_list = [all_items[i] for i in selected_indices]
        self.size = len(self.data_list)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        img_path, label = self.data_list[index]
        img = Image.open(img_path).convert('RGB')
        img_torch = self.transform(img)
        return img_torch, label

    def __len__(self):
        return self.size
