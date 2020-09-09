# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu


from torch.nn.init import normal_, constant_
from .basic_ops import ConsensusModule
from torch import nn
import torchvision


class TSM(nn.Module):
    def __init__(self,
                 args,  
                 num_class, 
                 input_size=224,
                 num_segments=8,
                 new_length=None,
                 img_feature_dim=256,
                 crop_num=1,
                 partial_bn=True,
                 print_spec=True,
                 pretrain='imagenet',
                 is_shift=False,
                 shift_div=8,
                 base_model='resnet50',
                 shift_place='blockres',
                 temporal_pool=False,
                 non_local=False):

        super(TSM, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.crop_num = crop_num
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.input_size = input_size

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if new_length is None:
            self.new_length = 1
        else:
            self.new_length = new_length
    
        self._prepare_base_model(self.base_model_name)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, x):
        """
        input x: B, N, C, H, W, ex: torch.Size([4, 8, 3, 224, 224])
        """
        # view to torch.Size([32, 3, 224, 224])
        x = x.view((-1, 3) + x.size()[-2:])

        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
   
        return x

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                from .temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                from .non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            del self.base_model.avgpool
            del self.base_model.fc

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def partialBN(self, enable):
        self._enable_pbn = enable
