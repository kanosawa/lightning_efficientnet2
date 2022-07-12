import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy


class EfficientNetModule(LightningModule):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.__dict__['efficientnet_v2_s'](pretrained=True).features
        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, num_class)
        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.base_model(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch):
        loss, accuracy = self.step(batch)
        self.log_dict({'train_loss': loss,
                       'train_accuracy': accuracy,
                       'step': torch.tensor(self.current_epoch, dtype=torch.float32)},
                      on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log_dict({'val_loss': loss,
                       'val_accuracy': accuracy,
                       'step': torch.tensor(self.current_epoch, dtype=torch.float32)},
                      on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def step(self, batch):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        accuracy = self.accuracy(output, y)
        return loss, accuracy


def main():

    num_epoch = 10
    batch_size = 128
    num_class = 10

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNetModule(num_class)
    model_checkpoint = ModelCheckpoint(monitor='val_loss', filename='{epoch:02d}', mode='min')
    trainer = Trainer(gpus=[1], max_epochs=num_epoch, callbacks=[model_checkpoint], deterministic=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    seed_everything(42)
    main()
