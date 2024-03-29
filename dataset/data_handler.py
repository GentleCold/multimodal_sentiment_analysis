import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from transformers import BertTokenizer

EMOTION2ID = {"positive": 0, "neutral": 1, "negative": 2, "null": 3}
ID2EMOTION = {0: "positive", 1: "neutral", 2: "negative", 3: "null"}
SEED = 42

dataset_loc = "./dataset"


class MultimodalDataset(IterableDataset):
    def __init__(self, samples, tokenizer):
        self.data = samples
        self.tokenizer = tokenizer
        self.samples = []

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
            self.samples.append((guid, txt, img, EMOTION2ID[tag]))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for bundles in self.samples:
            yield bundles


class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", mirror="bfsu"
        )
        self.get_dataloader()

    @staticmethod
    def _collate_fn(batch):
        guid = []
        txt = []
        img = []
        tag = []
        max_len = 0
        for sample in batch:
            guid.append(sample[0])
            txt.append(sample[1])
            max_len = max(max_len, len(sample[1]))
            img.append(np.array(sample[2]).tolist())
            tag.append(sample[3])

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
        return guid, padded_txt, txt_mask, img, tag

    def get_dataloader(self):
        print("===== Data Info =====")
        samples = pd.read_csv(f"{dataset_loc}/train.txt")
        print(samples["tag"].value_counts())

        train_samples, val_samples = train_test_split(
            samples, test_size=0.2, random_state=SEED
        )
        test_samples = pd.read_csv(f"{dataset_loc}/test_without_label.txt").fillna(
            "null"
        )

        self.train_size = len(train_samples)
        self.val_size = len(val_samples)
        self.test_size = len(test_samples)

        print("\nTrain size:", self.train_size)
        print("Val size:", self.val_size)
        print("Test size:", self.test_size, "\n")
        print("Loading dataset...\n")

        train_data = MultimodalDataset(train_samples, self.tokenizer)
        val_data = MultimodalDataset(val_samples, self.tokenizer)
        test_data = MultimodalDataset(test_samples, self.tokenizer)

        self.train_loader = DataLoader(
            train_data, batch_size=self.batch_size, collate_fn=self._collate_fn
        )
        self.val_loader = DataLoader(
            val_data, batch_size=self.batch_size, collate_fn=self._collate_fn
        )
        self.test_loader = DataLoader(
            test_data, batch_size=self.batch_size, collate_fn=self._collate_fn
        )
