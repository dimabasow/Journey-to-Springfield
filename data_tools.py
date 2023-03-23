# from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class SimpsonsDataSet(Dataset):
    def __init__(self, files, transform, labels=None):
        super().__init__()

        # self.images = [Image.open(file) for file in sorted(files)]
        self.images = [torchvision.io.read_image(str(file)) for file in sorted(files)]
        self.transform = transform
        self.labels = labels
        self.len_ = len(self.images)

        if labels is not None:
            self.getitem = self.getitem_train_val
        else:
            self.getitem = self.getitem_test

    def getitem_train_val(self, index):
        image_transformed = self.transform(self.images[index])
        label = self.labels[index]
        return image_transformed, label

    def getitem_test(self, index):
        image_transformed = self.transform(self.images[index])
        return image_transformed

    def __len__(self):
        return self.len_

    def __getitem__(self, item):
        return self.getitem(item)

class SimpsonsDataModule(pl.LightningDataModule):
    def __init__(self, path_to_train, path_to_test, val_size=0.25,
                 preprocess_transform=None, augmentation_transform=None, batch_size=32):
        super().__init__()
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.val_size = val_size
        self.preprocess_transform = preprocess_transform
        self.augmentation_transform = augmentation_transform
        self.batch_size = batch_size

        # Чтение путей и создание списокв файлов
        TRAIN_DIR = Path(self.path_to_train)
        TEST_DIR = Path(self.path_to_test)
        train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
        self.predict_files = sorted(list(TEST_DIR.rglob('*.jpg')))

        # Кодирования классов
        self.label_encoder = LabelEncoder()
        train_val_labels = [path.parent.name for path in train_val_files]
        train_val_labels_encoded = self.label_encoder.fit_transform(train_val_labels)

        # Разбивка на тренировочную и валидационную часть
        self.train_files, self.val_files, self.train_labels, self.val_labels = train_test_split(train_val_files,
                                                                                                train_val_labels_encoded,
                                                                                                test_size=self.val_size)

    def setup(self, stage):
        # Трансформация и аугментация
        if self.augmentation_transform is not None:
            augmentation_transform = nn.Sequential(self.augmentation_transform,self.preprocess_transform)
        else:
            augmentation_transform = self.preprocess_transform
        preprocess_transform = self.preprocess_transform

        # Создание датасетов
        self.train_dataset = SimpsonsDataSet(self.train_files, augmentation_transform, self.train_labels)
        self.val_dataset = SimpsonsDataSet(self.val_files, preprocess_transform, self.val_labels)
        self.predict_dataset = SimpsonsDataSet(self.predict_files, preprocess_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)