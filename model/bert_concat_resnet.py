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
        self.txt_linear = nn.Linear(768, 128)
        self.img_linear = nn.Linear(1000, 128)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, txt, txt_mask, image):
        img_out = self.img_model(image)
        img_out = self.img_linear(img_out)
        img_out = self.relu(img_out)

        txt_out = self.txt_model(input_ids=txt, attention_mask=txt_mask)  # type: ignore
        txt_out = txt_out.last_hidden_state[:, 0, :]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.txt_linear(txt_out)
        txt_out = self.relu(txt_out)

        last_out = torch.cat((txt_out, img_out), dim=-1)
        last_out = self.fc(last_out)
        return last_out
