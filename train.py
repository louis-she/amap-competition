import pandas as pd
import torch
import cv2
import os
import numpy as np
from matplotlib import pyplot
import albumentations
import sys
import minetorch
from radam import RAdam
import efficientnet_pytorch as enet
from tqdm.notebook import tqdm
from minetorch.metrics import MultiClassesClassificationMetricWithLogic
from warmup_scheduler import GradualWarmupScheduler
from minetorch.spreadsheet import GoogleSheet
import pretrainedmodels
from albumentations.pytorch import ToTensorV2
import nni
import sklearn
from nni.utils import merge_parameter

# parameters
fold = 3
batch_size = 16
num_workers = 4
init_lr = 0.01
n_epochs = 30
code = 'newbie-3'
accumulated_iter = 1
image_size = (180, 320)
data_sampler_weights = (1, 10, 5)

# environment related
csv = './flatten.csv'
image_dir = '/home/jovyan/data/amap_traffic_train_0712'

df = pd.read_csv(csv, dtype={'data_id': 'string'})
val_df = df[df.fold == fold].reset_index()
train_df = df[df.fold != fold].reset_index()

# transforms
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transforms_train = albumentations.Compose([
    albumentations.OneOf([
        albumentations.IAAAdditiveGaussianNoise(),
        albumentations.GaussNoise(),
    ], p=0.5),
    albumentations.OneOf([
        albumentations.CLAHE(clip_limit=2),
        albumentations.IAASharpen(),
        albumentations.IAAEmboss(),
        albumentations.RandomBrightnessContrast(),
    ], p=0.5),
    albumentations.HueSaturationValue(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Resize(*image_size),
    albumentations.Normalize(mean=mean, std=std, p=1),
    ToTensorV2()
])
transforms_val = albumentations.Compose([
    albumentations.Resize(*image_size),
    albumentations.Normalize(mean=mean, std=std, p=1),
    ToTensorV2()
])

# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.df.iloc[index]
        subdir = row.data_id
        image = cv2.imread(os.path.join(image_dir, subdir, str(row['frame_name'])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = self.transforms(image=image)['image']
        label = row['status']
        return augmented_image, torch.tensor(label)

    def __len__(self):
        return len(self.df)

model = enet.EfficientNet.from_pretrained('efficientnet-b5', num_classes=3)

weights = np.array(train_df.status)
weights[weights == 0] = data_sampler_weights[0]
weights[weights == 1] = data_sampler_weights[1]
weights[weights == 2] = data_sampler_weights[2]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_df))

train_dataset = Dataset(train_df, transforms_train)
val_dataset = Dataset(val_df, transforms_val)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers)

loss = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0.00003)

def scheduler_step(miner, **payload):
    scheduler.step(miner.current_epoch)

# metrics
class Metric(MultiClassesClassificationMetricWithLogic):

    def before_init(self):
        super().before_init()
        self.create_sheet_column('f1_score', 'F1 Score')

    def after_epoch_end(self, val_loss, **ignore):
        super().after_epoch_end(val_loss, **ignore)

        result = sklearn.metrics.f1_score(self.targets, self.predicts, average=None)
        f1 = np.sum(result * np.array([0.2, 0.2, 0.6]))

        png_file = self.scalars(
            {'weighted_sum': f1, 'class0': result[0], 'class1': result[1], 'class2': result[2]}, 'f1_score'
        )
        if png_file:
            self.update_sheet('f1_score', {'raw': png_file, 'processor': 'upload_image'})

miner = minetorch.Miner(
    alchemistic_directory='./',
    code=code,
    model=model.cuda(),
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_func=loss,
    drawer='matplotlib',
    gpu=True,
    max_epochs=n_epochs,
    plugins=[
        Metric(),
    ],
    hooks={
        'before_epoch_start': scheduler_step
    },
    sheet=GoogleSheet('1Rw2jgxLNe3QRkaN_bFPrjn0UX40DYAteY_abubpREps', 'quickstart.json', build_kwargs=dict(cache_discovery=False)),
    accumulated_iter=accumulated_iter,
    trival=False,
    ignore_optimizer_resume=True
)

miner.train()

