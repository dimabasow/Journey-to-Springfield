import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torchmetrics

class Classifier(pl.LightningModule):
    def __init__(self, model_name="efficientnet_v2_l", unfreezed_fraction=0.0, learning_rate=4e-3):
        super().__init__()
        self.save_hyperparameters()

        num_classes=42
        self.learning_rate = learning_rate

        weights = list(torchvision.models.get_model_weights(model_name))[0]
        self.backbone_model = torchvision.models.get_model(model_name, weights=weights)
        self.preprocess_transform = weights.transforms()

        self.freeze_update(unfreezed_fraction)

        in_features = self.backbone_model.classifier[-1].in_features
        self.backbone_model.classifier = nn.Linear(in_features, num_classes)

        self.loss_function = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def freeze_update(self, unfreezed_fraction):
        feature_extractor = self.backbone_model.features
        num_parameters = len(list(feature_extractor.parameters()))
        unfreezed_parameters = int(num_parameters*unfreezed_fraction)
        print("КОЛИЧЕСТВО ПАРАМЕТРОВ", num_parameters)
        print("КОЛИЧЕСТВО ОБУЧАЕМЫХ ПАРАМЕТРОВ", unfreezed_parameters)
        if unfreezed_parameters:
            for p in list(feature_extractor.parameters())[:-unfreezed_parameters]:
                p.requires_grad = False
            for p in list(feature_extractor.parameters())[-unfreezed_parameters:]:
                p.requires_grad = True
        else:
            for p in feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, x):
        logits = self.backbone_model(x)
        return logits


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        self.valid_acc(y_pred, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer