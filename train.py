import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from dataset.data_handler import EMOTION, SEED, get_dataloader
from model.bert_concat_resnet import BertConcatResnet

model_save_path = "./output/model.pt"

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
        learning_rate=0.001,
        batch_size=8,
        model="bert_concat_resnet",
    ):
        set_seed(SEED)

        self.epochs = max_epochs
        self.batch_size = batch_size
        if model == "bert_concat_resnet":
            self.model = BertConcatResnet().to(DEVICE)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.train_loader, self.val_loader = get_dataloader(batch_size)

        def initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.model.apply(initialize_weights)

    def train(self):
        print("===== Traning Info =====")
        print("Device:", DEVICE)
        print("Batch size:", self.batch_size)
        print("\n==== Starting Train ====")

        best_valid_loss = float("inf")
        early_stop_patience = 3
        early_stop_count = 0
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            self._epoch_train(epoch)
            valid_loss = self._evaluate(self.val_loader)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), model_save_path)
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count == early_stop_patience:
                    print("Early stop...")
                    break

        print(f"The training epoch is {epoch}")
        print(f"Choose model with best valid loss: {best_valid_loss}")
        self.model.load_state_dict(torch.load(model_save_path))

    def _epoch_train(self, epoch):
        self.model.train()
        epoch_loss = 0
        correct = 0
        for txt, txt_mask, image, label in tqdm(self.train_loader):
            txt = txt.to(DEVICE)
            txt_mask = txt_mask.to(DEVICE)
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            self.optimizer.zero_grad()
            output = self.model(txt, txt_mask, image)

            loss = self.criterion(output, label)
            loss.backward()

            # used to prevent gradient explosion
            clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()

            epoch_loss += loss.item()

        epoch_loss /= len(self.train_loader)
        epoch_acc = correct / len(self.train_loader)
        print(f"Train Epoch {epoch}")
        print("Train set: \nLoss: {}, Accuracy: {}".format(epoch_loss, epoch_acc))

    def _evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0
        correct = 0

        with torch.no_grad():
            for txt, txt_mask, image, label in loader:
                txt = txt.to(DEVICE)
                txt_mask = txt_mask.to(DEVICE)
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                output = self.model(txt, txt_mask, image)

                loss = self.criterion(output, label)

                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                epoch_loss += loss.item()

        epoch_loss /= len(loader)
        epoch_acc = correct / len(loader)
        print("Valid set: \nLoss: {}, Accuracy: {}".format(epoch_loss, epoch_acc))
        return epoch_loss
