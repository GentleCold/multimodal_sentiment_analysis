import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import optim
from tqdm import tqdm

from dataset.data_handler import ID2EMOTION, SEED, DataHandler
from model.bert_densenet_with_attention import BertDensenetWithAttention
from model.bert_densenet_with_concat import BertDensenetWithConcat
from model.bert_resnet_with_attention import BertResnetWithAttention
from model.bert_resnet_with_concat import BertResnetWithConcat

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
        max_epochs=10,
        lr=1e-5,
        batch_size=16,
        model=0,
        ablate=0,
    ):
        set_seed(SEED)

        self.epochs = max_epochs
        self.batch_size = batch_size
        self.ablate = ablate
        self.model_type = model
        if model == 0:
            self.model = BertResnetWithConcat().to(DEVICE)
        elif model == 1:
            self.model = BertResnetWithAttention().to(DEVICE)
        elif model == 2:
            self.model = BertDensenetWithConcat().to(DEVICE)
        elif model == 3:
            self.model = BertDensenetWithAttention().to(DEVICE)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.data = DataHandler(self.batch_size)

    def train(self):
        self.train_loss = []
        self.val_accuracy = []
        print("===== Traning Info =====")
        print("Device:", DEVICE)
        print("Batch size:", self.batch_size)

        if self.ablate == 0:
            if self.model_type == 0:
                print("Model: BertResnetWithConcat")
            elif self.model_type == 1:
                print("Model: BertResnetWithCrossAttention")
            elif self.model_type == 2:
                print("Model: BertDensenetWithConcat")
            elif self.model_type == 3:
                print("Model: BertDensenetWithCrossAttention")
        elif self.ablate == 1:
            if self.model_type == 0:
                print("Model: Resnet Only")
            elif self.model_type == 2:
                print("Model: Densenet Only")
        elif self.ablate == 2:
            print("Model: Bert only")

        print("\n==== Starting Train ====")

        best_metrics = [float("inf"), 0, 0, 0, 0]
        early_stop_patience = 3
        early_stop_count = 0
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            self._epoch_train(epoch)
            metrics = self._evaluate()
            if metrics[1] > best_metrics[1]:
                best_metrics = metrics
                torch.save(self.model.state_dict(), f"{output_loc}/model.pt")
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count == early_stop_patience:
                    print("Early stop...")
                    break

        print(f"The training epoch is {epoch}")
        print(f"Choose model with best accuracy: {best_metrics[1]}, with:")
        print(f"Precision: {best_metrics[2]}")
        print(f"Recall: {best_metrics[3]}")
        print(f"F1: {best_metrics[4]}")
        self.model.load_state_dict(torch.load(f"{output_loc}/model.pt"))

        def moving_average(data, window_size):
            weights = np.repeat(1.0, window_size) / window_size
            smoothed_data = np.convolve(data, weights, "valid")
            return smoothed_data

        window_size = 10
        self.train_loss = moving_average(self.train_loss, window_size)

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
            output = self.model(txt, txt_mask, image, self.ablate)

            loss = self.criterion(output, label)
            loss.backward()

            # used to prevent gradient explosion
            # clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()

            self.train_loss.append(loss.item())
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
                output = self.model(txt, txt_mask, image, self.ablate)

                loss = self.criterion(output, label)

                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                epoch_loss += loss.item()

                preds.append(pred)
                labels.append(label)

        epoch_loss /= len(self.data.val_loader)
        epoch_acc = correct / self.data.val_size
        self.val_accuracy.append(epoch_acc)

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
            for guid, txt, txt_mask, image, _ in tqdm(self.data.test_loader):
                txt = txt.to(DEVICE)
                txt_mask = txt_mask.to(DEVICE)
                image = image.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(txt, txt_mask, image, self.ablate)
                pred = output.argmax(dim=1)

                guid_list.extend(guid)
                emotion = [ID2EMOTION[p.item()] for p in pred]
                tag_list.extend(emotion)

        df = pd.DataFrame({"guid": guid_list, "tag": tag_list})
        df.to_csv(f"{output_loc}/test_with_predict.txt", index=False)
        print("Save successfully!")
