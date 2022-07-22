import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

# import backbone.get_maskrcnn_feature_extractor
# from detectron2.modeling.backbone import Backbone
# from maskformer_backbone import get_backbone
from MaskFormer.demo.demo import get_maskformer
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.trg_vocab_size = 38
        self.backbone = get_maskformer().backbone
        # print(self.backbone.state_dict()['res5.2.conv3.weight'][0][0])
        self.fc1 = nn.Sequential(nn.Linear(2048,300))
        self.en_lstm = nn.LSTM(input_size=900, hidden_size=512, num_layers=1, batch_first=True)
        self.de_lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.trg_vocab_size)

    def forward(self, x_3d_front, x_3d_right, x_3d_left, trg):
        hidden = None
        for t in range(x_3d_front.size(1)):
            with torch.no_grad():
                # print(self.backbone(x_3d_front[:, t, :, :, :]).keys())
                x = self.backbone(x_3d_front[:, t, :, :, :])['res5']
                print(x.shape)
                x = self.fc1(x)
                x_left = self.fc1(self.backbone(x_3d_left[:, t, :, :, :])['res5'])
                x_right = self.fc1(self.backbone(x_3d_right[:, t, :, :, :])['res5']) 
                x = torch.cat((x, x_left, x_right), dim=1)
            _, hidden = self.en_lstm(x.unsqueeze(1), hidden)

        out = self.fc3(self.fc2(hidden))
        return out

a = torch.zeros([2, 3, 3, 400, 200], dtype=torch.float32).cuda()
b = torch.zeros([2, 3, 3, 400, 200], dtype=torch.float32).cuda()
c = torch.zeros([2, 3, 3, 400, 200], dtype=torch.float32).cuda()
gt = torch.zeros([2, 1], dtype=torch.float32)

model = CNNLSTM(10).cuda()
out = model(a, b, c, gt) 
        

        