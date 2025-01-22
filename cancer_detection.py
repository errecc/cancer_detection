from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import opendatasets as od
import os
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from PIL import Image
from pprint import pprint
import numpy as np


class CancerTypesDataset(Dataset):
    def __init__(self, path):
        dirs = os.listdir(path)
        paths = [os.path.join(path, p) for p in dirs]
        labels = [{"path": p, "label": os.listdir(p)} for p in paths]
        label_data = []
        for p in labels:
            for l in p["label"] :
                label = l
                path = p["path"]
                data = {
                        "label": l,
                        "path": path,
                        "files": os.listdir(os.path.join(path,l))
                        }
                label_data.append(data)
        all_data = []
        all_labels = []
        for d in label_data:
            for f in d["files"]:
                label = d["label"]
                file_path = os.path.join(d["path"], label, f)
                all_labels.append(label)
                data = {
                        "file_path": file_path,
                        "label": label
                        }
                all_data.append(data)
        # Just get the labels in numerical way
        ls = {}
        for j,i in enumerate(list(set(all_labels))):
            ls[i] = j
        self.data = []
        self.out_features = len(ls)
        for d in all_data:
            data = {
                    "string_label": d["label"],
                    "label": ls[d["label"]],
                    "file_path": d["file_path"]
                    }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Label
        label = self.data[idx]["label"]
        labels = []
        for i in range(self.out_features):
            if i == label:
                labels.append(torch.tensor(1.))
            else:
                labels.append(torch.tensor(0.))
        labels = torch.tensor(labels, dtype = torch.float32).flatten(0)
        label = torch.tensor(label)
        # Image
        image = Image.open(self.data[idx]["file_path"])
        tensor = torch.tensor(np.array(image), dtype = torch.float32)
        tensor = tensor.reshape([3,512,512])
        return tensor, labels.flatten()


class CancerPredictionModel(pl.LightningModule):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3 , out_channels=16, kernel_size=10 ),
                torch.nn.MaxPool2d(4),
                torch.nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size = 5),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(0),
                torch.nn.Linear(14400, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512,256),
                torch.nn.ReLU(),
                torch.nn.Linear(256,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,32),
                torch.nn.ReLU(),
                torch.nn.Linear(32,out_features),
                )

    def forward(self,x):
        outs = self.model(x)
        return outs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        inp, lab = batch
        out = self(inp)
        lab = lab.flatten()
        loss = self.loss_fn(out, lab)
        self.log(f"t_loss", loss, on_step=True, on_epoch=True)
        return loss


if __name__ == "__main__":
    # Download the dataset
    cancer_url = 'https://www.kaggle.com/datasets/obulisainaren/multi-cancer' 
    od.download(cancer_url)
    path = os.path.join("multi-cancer", "Multi Cancer", "Multi Cancer")

    # main training from lightning
    dataset = CancerTypesDataset(path)
    dataxd = dataset[0]
    model = CancerPredictionModel(dataset.out_features)
    loader = DataLoader(dataset, num_workers=3)
    trainer = pl.Trainer(max_epochs = 100)
    trainer.fit(model, loader)
