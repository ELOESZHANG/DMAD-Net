import torch.nn as nn
import torch.nn.functional as F
from fusion import FreqFusion

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        # self.fc1=nn.Linear(312,88)
        # self.fc2 = nn.Conv1d(100,100,1)
        # self.fc3 = nn.Conv2d(16,256,3,1,1)
        #
        # self.fusion = FreqFusion(hr_channels=256, lr_channels=256)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        # x_image = batch_dict['image_features']
        # target_shape = (1, 16, 100, 312)
        # current_shape = x_image.shape
        #
        # # 计算填充的数量
        # pad_width = target_shape[-1] - current_shape[-1]  # 312 - 300 = 12
        # pad_length = target_shape[-2] - current_shape[-2]
        #
        # # 在最后一个维度填充
        # # pad 的格式是 (左边填充, 右边填充)
        # x_image = F.pad(x_image, (0, pad_width, 0, 0, 0, 0, 0, 0))
        # x_image = F.pad(x_image, (0, 0, 0, pad_length), mode='constant', value=0)
        #
        #
        # N, C, H, W = x_image.shape
        # x_image= self.fc1(x_image.view(-1,W)).reshape(N*C,H,-1)
        # x_image=self.fc2(x_image).reshape(N,C,100,-1)
        # x_image = self.fc3(x_image)
        #
        # _,spatial_features_0,_ = self.fusion(spatial_features,x_image)
        #
        # batch_dict['spatial_features'] =F.normalize(spatial_features_0,dim=-1)  + spatial_features
        batch_dict['spatial_features'] =spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
