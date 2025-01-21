from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import opendatasets as od
import os
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
        image = Image.open(self.data[idx]["file_path"])
        label = self.data[idx]["label"]
        tensor = torch.tensor(np.array(image))
        return tensor, label


path = os.path.join("multi-cancer", "multi", "multi")
print(path)
data = CancerTypesDataset(path)
# Download the dataset
cancer_url = 'https://www.kaggle.com/datasets/obulisainaren/multi-cancer' 
od.download(cancer_url)
