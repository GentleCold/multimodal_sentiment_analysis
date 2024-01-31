import torch
import torch.nn as nn
import torchvision
from torchvision.models import DenseNet121_Weights
from transformers import BertModel


class BertDensenetWithConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained("bert-base-uncased")
        self.img_model = torchvision.models.densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        self.fc = nn.Linear(768 + 1000, 3)
        self.only_img_fc = nn.Linear(1000, 3)
        self.only_txt_fc = nn.Linear(768, 3)

    def forward(self, txt, txt_mask, img, ablate):
        if ablate == 0:  # both
            img = self.img_model(img)
            txt = self.txt_model(input_ids=txt, attention_mask=txt_mask)  # type: ignore
            txt = txt.last_hidden_state[:, 0, :]  # CLS vector
            txt.view(txt.shape[0], -1)

            out = torch.cat((txt, img), dim=-1)
            out = self.fc(out)
        elif ablate == 1:  # img only
            img = self.img_model(img)
            out = self.only_img_fc(img)
        else:  # txt only
            txt = self.txt_model(input_ids=txt, attention_mask=txt_mask)  # type: ignore
            txt = txt.last_hidden_state[:, 0, :]  # CLS vector
            txt.view(txt.shape[0], -1)
            out = self.only_txt_fc(txt)

        return out
