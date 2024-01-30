import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from transformers import BertModel


class BertConcatResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained("bert-base-uncased")
        self.img_model = torchvision.models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        self.txt_fc = nn.Linear(768, 256)
        self.img_fc = nn.Linear(1000, 256)

        self.fc = nn.Linear(512, 3)
        self.only_fc = nn.Linear(256, 3)
        self.activate = nn.LeakyReLU()

    def forward(self, txt, txt_mask, img, ablate):
        if ablate == 0:  # both
            img = self.img_model(img)
            img = self.img_fc(img)
            img = self.activate(img)

            txt = self.txt_model(input_ids=txt, attention_mask=txt_mask)  # type: ignore
            txt = txt.last_hidden_state[:, 0, :]  # CLS vector
            txt.view(txt.shape[0], -1)
            txt = self.txt_fc(txt)
            txt = self.activate(txt)

            out = torch.cat((txt, img), dim=-1)
            out = self.fc(out)
        elif ablate == 1:  # img only
            img = self.img_model(img)
            img = self.img_fc(img)
            img = self.activate(img)
            out = self.only_fc(img)
        else:  # txt only
            txt = self.txt_model(input_ids=txt, attention_mask=txt_mask)  # type: ignore
            txt = txt.last_hidden_state[:, 0, :]  # CLS vector
            txt.view(txt.shape[0], -1)
            txt = self.txt_fc(txt)
            txt = self.activate(txt)
            out = self.only_fc(txt)

        return out
