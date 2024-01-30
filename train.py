import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from dataset.data_handler import ID2EMOTION, SEED, DataHandler
from model.bert_concat_resnet import BertConcatResnet

output_loc = "./output"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Model:
    def __init__(
        self,
        max_epochs=30,
        learning_rate=1e-5,
        batch_size=8,
        model="bert_concat_resnet",
    ):
        set_seed(SEED)

        self.epochs = max_epochs
        self.batch_size = batch_size
        if model == "bert_concat_resnet":
            self.model = BertConcatResnet().to(DEVICE)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.data = DataHandler(self.batch_size)

    def train(self):
        print("===== Traning Info =====")
        print("Device:", DEVICE)
        print("Batch size:", self.batch_size)
        print("\n==== Starting Train ====")

        best_metrics = [float("inf")]
        early_stop_patience = 3
        early_stop_count = 0
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            self._epoch_train(epoch)
            metrics = self._evaluate()
            if metrics[0] < best_metrics[0]:
                best_metrics = metrics
                torch.save(self.model.state_dict(), f"{output_loc}/model.pt")
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count == early_stop_patience:
                    print("Early stop...")
                    break

        print(f"The training epoch is {epoch}")
        print(f"Choose model with best valid loss: {best_metrics[0]}, with:")
        print(f"Accuracy: {best_metrics[1]}")
        print(f"Precision: {best_metrics[2]}")
        print(f"Recall: {best_metrics[3]}")
        print(f"F1: {best_metrics[4]}")
        self.model.load_state_dict(torch.load(f"{output_loc}/model.pt"))

    def _epoch_train(self, epoch):
        self.model.train()
        epoch_loss = 0
        correct = 0
        for _, txt, txt_mask, image, label in tqdm(self.data.train_loader):
            txt = txt.to(DEVICE)
            txt_mask = txt_mask.to(DEVICE)
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            self.optimizer.zero_grad()
            output = self.model(txt, txt_mask, image)

            loss = self.criterion(output, label)
            loss.backward()

            # used to prevent gradient explosion
            # clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()

            epoch_loss += loss.item()

        epoch_loss /= len(self.data.train_loader)
        epoch_acc = correct / self.data.train_size
        print(f"Train Epoch {epoch}")
        print("Train set: \nLoss: {}, Accuracy: {}".format(epoch_loss, epoch_acc))

    def _evaluate(self):
        self.model.eval()
        epoch_loss = 0
        correct = 0
        preds = []
        labels = []

        with torch.no_grad():
            for _, txt, txt_mask, image, label in self.data.val_loader:
                txt = txt.to(DEVICE)
                txt_mask = txt_mask.to(DEVICE)
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(txt, txt_mask, image)

                loss = self.criterion(output, label)

                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                epoch_loss += loss.item()

                preds.append(pred)
                labels.append(label)

        epoch_loss /= len(self.data.val_loader)
        epoch_acc = correct / self.data.val_size

        preds = torch.cat(preds).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        f1 = f1_score(labels, preds, average="weighted")
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds, average="weighted")
        print(
            "Valid set: \nLoss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(
                epoch_loss, epoch_acc, precision, recall, f1
            )
        )
        return epoch_loss, epoch_acc, precision, recall, f1

    def save_test_result(self):
        print(f"Predict and save test result...")
        self.model.eval()
        guid_list = []
        tag_list = []

        with torch.no_grad():
            for guid, txt, txt_mask, image, label in tqdm(self.data.test_loader):
                txt = txt.to(DEVICE)
                txt_mask = txt_mask.to(DEVICE)
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(txt, txt_mask, image)
                pred = output.argmax(dim=1)

                guid_list.extend(guid)
                emotion = [ID2EMOTION[p.item()] for p in pred]
                tag_list.extend(emotion)

        df = pd.DataFrame({"guid": guid_list, "tag": tag_list})
        df.to_csv(f"{output_loc}/test_with_predict.txt", index=False)
        print("Save successfully!")
