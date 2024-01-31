import torch
import torch.nn as nn
import torchvision
from torchvision.models import DenseNet121_Weights
from transformers import BertModel


class BertDensenetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained("bert-base-uncased")
        self.img_model = torchvision.models.densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        self.txt_fc = nn.Linear(768, 256)
        self.img_fc = nn.Linear(1000, 256)

        self.fc = nn.Linear(512, 3)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512,
                nhead=4,
                dim_feedforward=512,
                dropout=0.4,
            ),
            num_layers=2,
        )

        self.activate = nn.LeakyReLU()

    def forward(self, txt, txt_mask, img, _):
        img = self.img_model(img)
        img = self.img_fc(img)
        img = self.activate(img)

        txt = self.txt_model(input_ids=txt, attention_mask=txt_mask)  # type: ignore
        txt = txt.last_hidden_state[:, 0, :]  # CLS vector
        txt.view(txt.shape[0], -1)
        txt = self.txt_fc(txt)
        txt = self.activate(txt)

        out = torch.cat((txt, img), dim=-1)
        out = self.transformer_encoder(out)
        out = self.fc(out)

        return out
