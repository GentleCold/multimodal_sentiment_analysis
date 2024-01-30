import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from transformers import BertTokenizer

EMOTION = {"positive": 0, "neutral": 1, "negative": 2}
SEED = 42

dataset_loc = "./dataset"


class MultimodalDataset(IterableDataset):
    def __init__(self, samples, tokenizer):
        self.data = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for guid, tag in zip(self.data["guid"], self.data["tag"]):
            # load img
            img = Image.open(f"{dataset_loc}/data/{guid}.jpg")
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
            img = transform(img)

            # load txt
            with open(f"{dataset_loc}/data/{guid}.txt", "r", encoding="gb18030") as f:
                txt = f.read()
                txt = txt.replace("#", "")

            txt = self.tokenizer.encode(txt, add_special_tokens=True)

            yield txt, img, EMOTION[tag]


def _collate_fn(batch):
    txt = [sample[0] for sample in batch]
    img = [np.array(sample[1]).tolist() for sample in batch]
    tag = [sample[2] for sample in batch]

    max_len = max(len(t) for t in txt)
    padded_txt = []
    txt_mask = []
    for t in txt:
        padded = t + [0] * (max_len - len(t))
        mask = [1] * len(t) + [0] * (max_len - len(t))
        padded_txt.append(padded)
        txt_mask.append(mask)

    padded_txt = torch.LongTensor(padded_txt)
    txt_mask = torch.BoolTensor(txt_mask)
    img = torch.FloatTensor(img)
    tag = torch.LongTensor(tag)
    return padded_txt, txt_mask, img, tag


def get_dataloader(batch_size):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", mirror="bfsu")
    samples = pd.read_csv(f"{dataset_loc}/train.txt")
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, random_state=SEED
    )

    train_data = MultimodalDataset(train_samples, tokenizer)
    val_data = MultimodalDataset(val_samples, tokenizer)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=_collate_fn)

    return train_loader, val_loader
