from model.backbone.tsm.models import TSM
import torch.nn.functional as F
from torch import nn
import torch


class Basketball_Highlight_Generator(nn.Module):
    def __init__(self, 
                 tsm_pretrain_path='access https://github.com/mit-han-lab/temporal-shift-module to choose pretrain',
                 hidden_layer=[512, 64, 8],
                 arch_temporal='resnet50',
                 image_size=224,
                 num_segments=8, 
                 shift_div=8,
                 num_class=2):
        super(Basketball_Highlight_Generator, self).__init__()

        self.hidden_layer = hidden_layer
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.num_class = num_class
        self.image_size = image_size

        # Define model's layers
        self.flatten_layer = nn.Sequential(nn.MaxPool2d((7, 7)))
        self.maxpool_layer = nn.Sequential(nn.MaxPool1d(8))

        # Build TSM Backbone
        self.temporal_backbone = TSM(self.num_class, self.num_segments, base_model=arch_temporal, is_shift=True, input_size=self.image_size)

        # Load pretrain, optional map_location
        if tsm_pretrain_path is not None:
            pretrain = torch.load(tsm_pretrain_path, map_location=torch.device('cuda:0'))['state_dict']
            pretrain.pop('module.new_fc.weight', None)
            pretrain.pop('module.new_fc.bias', None)
            
            self.temporal_backbone = nn.DataParallel(self.temporal_backbone)
            self.temporal_backbone.load_state_dict(pretrain)
            self.temporal_backbone = self.temporal_backbone.module

        self.layers = hidden_layer
        self.fc = nn.ModuleList()
        self.drop = nn.ModuleList()

        # Connect output of image's extractor vs first hidden layer
        fc = nn.Linear(2048, self.layers[0], bias=True)
        nn.init.kaiming_uniform_(fc.weight, nonlinearity='relu')
        self.fc.append(fc)
        self.drop.append(nn.Dropout(p=0.5))

        for size_current, size_next in zip(self.layers[:-1], self.layers[1:]):
            fc = nn.Linear(size_current, size_next, bias=True)
            nn.init.kaiming_uniform_(fc.weight, nonlinearity='relu')

            self.fc.append(fc)
            self.drop.append(nn.Dropout(p=0.5))

        # Connect output of hidden layer vs last layer(2 node)
        self.fc_last = nn.Linear(self.layers[-1], 2, bias=True)
        nn.init.kaiming_uniform_(self.fc_last.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Parameters:
            x: input, shape: (B, N, C, H, W), ex: torch.Size([4, 8, 3, 224, 224])
        Output: 
            fc_last values, shape: (B, 2), ex: torch.Size(4, 2)
        """
        B, N, _, _, _ = x.size()
        
        x = self.temporal_backbone(x)
        x = self.flatten_layer(x)

        x = x.view(x.size()[:2])
        x = x.view((-1, N) + x.size()[-1:])

        x = self.maxpool_layer(x)
        x = x.view(B, -1)

        for drop, fc in zip(self.drop, self.fc):
            x = drop(F.relu(fc(x)))
        
        return self.fc_last(x)

