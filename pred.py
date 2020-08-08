import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import minetorch
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations
from albumentations.pytorch import ToTensorV2
import efficientnet_pytorch as enet
from collections import Counter


image_dir = '/home/jovyan/data/amap_traffic_test_0712'
df = pd.read_csv('/home/jovyan/work/amap/flatten_test.csv', dtype={'data_id': 'string'})

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = albumentations.Compose([
    albumentations.Resize(270, 480),
    albumentations.Normalize(mean=mean, std=std, p=1),
    ToTensorV2()
])


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
        return augmented_image, subdir

    def __len__(self):
        return len(self.df)

checkpoints = [
    '/home/jovyan/work/amap/epoch-20-270x480-bce@1:5-level3/models/epoch_11.pth.tar',
]

models = []
for checkpoint in checkpoints:
    model = enet.EfficientNet.from_pretrained('efficientnet-b0', num_classes=2).cuda()
    stuff = torch.load(checkpoint)
    model.load_state_dict(stuff['state_dict'])
    model.eval()
    models.append(model)

dataset = Dataset(df, transform)
test_loader = DataLoader(dataset, batch_size=1, num_workers=0)

# predict
ids = []
preds = []
with torch.no_grad():
    for data, data_id in tqdm(test_loader):
        logit = torch.zeros([2])
        for model in models:
            logit += model(data.cuda()).squeeze().detach().cpu()
        logit = logit / len(models)
        pred = logit.sigmoid().sum().round().numpy()
        preds.append(pred.astype(np.int))
        ids.append(data_id)

tmp = {}
for i,id in enumerate(ids):
    if id[0] not in tmp.keys():
        tmp[id[0]] = []
        tmp[id[0]].append(np.sum(preds[i]))
    else:
        tmp[id[0]].append(np.sum(preds[i]))

submits = []
for i,j in tmp.items():
    res = Counter(tmp[i]).most_common(3)
    if len(res)>1:
        if res[0][1] == res[1][1]:
            submits.append(res[1][0])
        else:
            submits.append(res[0][0])
    else:
         submits.append(res[0][0])

Counter(submits).most_common(3)

def transfer(annos, preds):
    for i, anno in enumerate(annos):
        anno['status'] = int(np.sum(preds[i]))

data = json.load(open('/home/jovyan/data/amap_traffic_annotations_test.json'))
transfer(data['annotations'], submits)

with open('/home/jovyan/data/submit_08_07_ens3.json', 'w') as f:
    json.dump(data, f)

