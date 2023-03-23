from data_tools import SimpsonsDataModule
from model import Classifier
import torch
import torchvision.transforms.v2 as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(42)
N_EPOCHS=100
BATCH_SIZE=32

model = Classifier()
augmentator = torch.nn.Sequential(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.0)),
                                  transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                  transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
                                  transforms.RandomRotation(degrees=45))
data_module = SimpsonsDataModule('simpsons_dataset', 'test', preprocess_transform=model.preprocess_transform, augmentation_transform=augmentator, batch_size=BATCH_SIZE)

#%%
logger = TensorBoardLogger("lightning_logs", name="Simpsons_classifier")
checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="valid_acc", save_last=True, every_n_epochs=1, mode="max")
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=N_EPOCHS,
)


trainer.fit(model=model, datamodule=data_module)


