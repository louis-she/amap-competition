import pandas as pd
import torch
import nni
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

params = nni.get_next_parameter()

# parameters
fold = 3
batch_size = 16
num_workers = 4
init_lr = 0.0003
n_epochs = 10
accumulated_iter = 1
data_sampler_weights = (1, 10, 5)

# nni parameters
image_size = params.get('input_size', '180x320')
loss_type = params.get('loss_type', 'ce@1:5:5')
aug_level = params.get('aug_level', 'level1')

code = f'{image_size}-{loss_type}-{aug_level}'

image_size = [int(x) for x in image_size.split('x')]

# environment related
csv = './flatten.csv'
image_dir = '/home/jovyan/data/amap_traffic_train_0712'

df = pd.read_csv(csv, dtype={'data_id': 'string'})
val_df = df[df.fold == fold].reset_index()
train_df = df[df.fold != fold].reset_index()

# transforms
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_levels = {
    'level1':   albumentations.Compose([
                    albumentations.RandomBrightness(),
                    albumentations.HorizontalFlip(),

                    albumentations.Resize(*image_size),
                    albumentations.Normalize(mean=mean, std=std, p=1),
                    ToTensorV2()
                ]),
    'level2':   albumentations.Compose([
                    albumentations.RandomBrightness(),
                    albumentations.Rotate(limit=(30, 30)),
                    albumentations.RGBShift(),
                    albumentations.HorizontalFlip(),
                    albumentations.RandomContrast(),

                    albumentations.Resize(*image_size),
                    albumentations.Normalize(mean=mean, std=std, p=1),
                    ToTensorV2()
                ]),
    'level3':   albumentations.Compose([
                    albumentations.RandomBrightness(),
                    albumentations.Rotate(limit=(30, 30)),
                    albumentations.RGBShift(),
                    albumentations.RandomGamma(),
                    albumentations.OpticalDistortion(),
                    albumentations.HorizontalFlip(),
                    albumentations.ShiftScaleRotate(),
                    albumentations.HueSaturationValue(),
                    albumentations.RandomContrast(),

                    albumentations.Resize(*image_size),
                    albumentations.Normalize(mean=mean, std=std, p=1),
                    ToTensorV2()
                ]),
    'level4':   albumentations.Compose([
                    albumentations.RandomBrightness(),
                    albumentations.Rotate(limit=(30, 30)),
                    albumentations.RGBShift(),
                    albumentations.RandomGamma(),
                    albumentations.ElasticTransform(),
                    albumentations.OpticalDistortion(),
                    albumentations.HorizontalFlip(),
                    albumentations.ShiftScaleRotate(),
                    albumentations.HueSaturationValue(),
                    albumentations.RandomContrast(),
                    albumentations.IAAAdditiveGaussianNoise(),

                    albumentations.Resize(*image_size),
                    albumentations.Normalize(mean=mean, std=std, p=1),
                    ToTensorV2()
                ]),
}

transforms_val = albumentations.Compose([
    albumentations.Resize(*image_size),
    albumentations.Normalize(mean=mean, std=std, p=1),
    ToTensorV2()
])

transforms_train = transform_levels[aug_level]

# metrics
class BaseMetric(MultiClassesClassificationMetricWithLogic):

    def before_init(self):
        super().before_init()
        self.current_f1 = None
        self.create_sheet_column('f1_score', 'F1 Score')

    def after_epoch_end(self, val_loss, **ignore):
        super().after_epoch_end(val_loss, **ignore)

        result = sklearn.metrics.f1_score(self.targets, self.predicts, average=None)
        self.current_f1 = np.sum(result * np.array([0.2, 0.2, 0.6]))

        png_file = self.scalars(
            {'weighted_sum': self.current_f1, 'class0': result[0], 'class1': result[1], 'class2': result[2]}, 'f1_score'
        )
        nni.report_intermediate_result(self.current_f1)
        if png_file:
            self.update_sheet('f1_score', {'raw': png_file, 'processor': 'upload_image'})

    def before_quit(self):
        nni.report_final_result(self.current_f1)


class RegressionMetric(BaseMetric):

    def after_val_iteration_ended(self, predicts, data, **ignore):
        """
        pred < 0.5:         0
        0.5 < pred < 1.5:   1
        ...
        pred > 4.5:         5
        """
        targets = data[1]
        predicts = predicts.detach().cpu().numpy().reshape([-1])
        predicts = np.ceil(predicts - 0.5).astype(np.int8)
        predicts[predicts < 0] = 0
        predicts[predicts > 2] = 2
        targets = targets.cpu().numpy().reshape([-1]).astype(np.int8)

        self.predicts = np.concatenate((self.predicts, predicts))
        self.targets = np.concatenate((self.targets, targets))


class BceMetric(BaseMetric):

    def after_val_iteration_ended(self, predicts, data, **ignore):
        targets = data[1]
        targets = targets.cpu().numpy().reshape([-1]).astype(np.int8)
        predicts = predicts.sigmoid().sum(1).detach().round().cpu().numpy()

        predicts[predicts < 0] = 0
        predicts[predicts > 2] = 2

        self.predicts = np.concatenate((self.predicts, predicts))
        self.targets = np.concatenate((self.targets, targets))



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
        return augmented_image, torch.tensor(row['status'])

    def __len__(self):
        return len(self.df)


if loss_type[:3] == 'bce':
    model = enet.EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    weights = [float(i) for i in loss_type.split('@')[1].split(':')]
    bce_loss = torch.nn.BCEWithLogitsLoss(torch.tensor(weights).cuda())
    def loss(data, targets):
        bce_targets = torch.zeros(len(targets), 2).cuda()
        for index, target in enumerate(targets.long()):
            bce_targets[index, :target] = 1
        return bce_loss(data, bce_targets)
    metric = BceMetric()
elif loss_type[:2] == 'ce':
    model = enet.EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
    weights = [float(i) for i in loss_type.split('@')[1].split(':')]
    loss = torch.nn.CrossEntropyLoss(torch.tensor(weights).cuda())
    metric = BaseMetric()
elif loss_type == 'reg':
    model = enet.EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
    l1_loss = torch.nn.SmoothL1Loss()
    def loss(data, targets):
        return l1_loss(data.squeeze(), targets.float())
    metric = RegressionMetric()


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

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=init_lr / 10)

def scheduler_step(miner, **payload):
    scheduler.step(miner.current_epoch)


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
    plugins=[ metric ],
    hooks={
        'before_epoch_start': scheduler_step
    },
    sheet=GoogleSheet('1Rw2jgxLNe3QRkaN_bFPrjn0UX40DYAteY_abubpREps', 'quickstart.json', build_kwargs=dict(cache_discovery=False)),
    accumulated_iter=accumulated_iter,
    trival=False,
    ignore_optimizer_resume=True
)

miner.train()

