import torch
import torch.nn.functional as F
import torch.nn as nn
from pcdet.utils.spconv_utils import spconv
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import split_voxels, check_repeat, FocalLoss
from pcdet.utils import common_utils


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.inchannel = 16
        self.depth = 8
        self.conv = nn.Conv2d(self.inchannel, self.depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(self.inchannel, self.depth, 1, 1)  # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(self.inchannel, self.depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(self.inchannel, self.depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(self.inchannel, self.depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(self.depth * 5, self.inchannel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)  # 池化分支
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))  # 汇合所有尺度的特征，利用1X1卷积融合特征输出
        return net


class SE(nn.Module):
    # ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)

        # relu激活，可自行换别的激活函数
        self.relu = nn.ReLU()

        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)

        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)

        x = self.relu(x)

        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)

        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


# 空间注意力机制
class spatial_attention(nn.Module):
    # 卷积核大小为7*7
    def __init__(self, kernel_size=7):
        super().__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2

        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)

        # 空间权重归一化
        x = self.sigmoid(x)

        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


# 通道注意力机制
class channel_attention(nn.Module):
    # ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍（可以换成1x1的卷积，效果相同）
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数（可以换成1x1的卷积，效果相同）
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()

        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)

        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        # （可以换成1x1的卷积，效果相同）
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool

        # sigmoid函数权值归一化
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Define the encoder blocks
        self.enc1 = self.double_conv(n_channels, 32)
        self.enc2 = self.double_conv(32, 64)
        self.enc3 = self.double_conv(64, 128)
        self.enc4 = self.double_conv(128, 256)
        # Define the max pooling layer
        self.maxpool = nn.MaxPool2d(2)
        # Define the decoder blocks
        self.dec1 = self.up_conv(256, 128)
        self.dec2 = self.up_conv(128, 64)
        self.dec3 = self.up_conv(64, 32)
        # Define the output layer
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        # Define a block of two convolutional layers with batch normalization and ReLU activation
        conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        return conv

    def up_conv(self, in_ch, out_ch):
        # Define a block of one up-convolutional layer and one double convolutional block
        upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        return upconv

    def forward(self, x):
        # Forward pass of the network# Encoding part
        b, c, h, w = x.shape
        x = torch.nn.functional.interpolate(x, (384, 1248), mode='bilinear', align_corners=False)
        x1 = self.enc1(x)
        x2 = self.maxpool(x1)
        x3 = self.enc2(x2)
        x4 = self.maxpool(x3)
        x5 = self.enc3(x4)
        x6 = self.maxpool(x5)
        x7 = self.enc4(x6)
        # Decoding part

        x8 = self.dec1(x7) + x5
        x9 = self.dec2(x8) + x3
        x10 = self.dec3(x9) + x1
        # Output layer
        output = self.outc(x10)
        aa = (384 - h) // 2
        bb = (1248 - w) // 2
        output = output[:, :, aa:h + aa, bb:w + bb]
        return output


class FocalSparseConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride, norm_fn=None, indice_key=None,
                 image_channel=3, kernel_size=3, padding=1, mask_multi=False, use_img=False,
                 topk=False, threshold=0.5, skip_mask_kernel=False, enlarge_voxel_channels=-1,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size=[0.1, 0.05, 0.05]):
        super(FocalSparseConv, self).__init__()

        self.conv = spconv.SubMConv3d(inplanes, planes, kernel_size=kernel_size, stride=1, bias=False,
                                      indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        offset_channels = kernel_size ** 3

        self.topk = topk
        self.threshold = threshold
        self.voxel_stride = voxel_stride
        self.focal_loss = FocalLoss()
        self.mask_multi = mask_multi
        self.skip_mask_kernel = skip_mask_kernel
        self.use_img = use_img

        voxel_channel = enlarge_voxel_channels if enlarge_voxel_channels > 0 else inplanes
        in_channels = image_channel + voxel_channel if use_img else voxel_channel

        self.conv_enlarge = spconv.SparseSequential(spconv.SubMConv3d(inplanes, enlarge_voxel_channels,
                                                                      kernel_size=3, stride=1, padding=1, bias=False,
                                                                      indice_key=indice_key + '_enlarge'),
                                                    norm_fn(enlarge_voxel_channels),
                                                    nn.ReLU(True)) if enlarge_voxel_channels > 0 else None

        self.conv_imp = spconv.SubMConv3d(in_channels, offset_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                          indice_key=indice_key + '_imp')

        _step = int(kernel_size // 2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step + 1) for j in range(-_step, _step + 1) for k in
                          range(-_step, _step + 1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda()
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        # self.ASPP = ASPP()
        # self.SE = SE(in_channel=16,ratio=4)
        # self.spatial_attention = spatial_attention(kernel_size=7)
        # self.channel_attention = channel_attention(in_channel=16,ratio=4)
        self.UNet = UNet(n_channels=16, n_classes=16)

    def construct_multimodal_features(self, x, x_rgb, batch_dict, fuse_sum=False):
        """
            Construct the multimodal features with both lidar sparse features and image features.
            Args:
                x: [N, C] lidar sparse features
                x_rgb: [b, c, h, w] image features
                batch_dict: input and output information during forward
                fuse_sum: bool, manner for fusion, True - sum, False - concat

            Return:
                image_with_voxelfeatures: [N, C] fused multimodal features
        """
        batch_index = x.indices[:, 0]
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]

        if not x_rgb.shape == batch_dict['images'].shape:
            x_rgb = nn.functional.interpolate(x_rgb, (h, w), mode='bilinear')

        image_with_voxelfeatures = []
        voxels_2d_int_list = []
        filter_idx_list = []
        # x_rgb = self.ASPP(x_rgb)
        # x_rgb =self.SE(x_rgb)
        ####together####
        # x_rgb = self.spatial_attention(x_rgb)
        # x_rgb = self.channel_attention(x_rgb)
        x_rgb = self.UNet(x_rgb)

        for b in range(batch_size):
            x_rgb_batch = x_rgb[b]

            calib = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index == b]
            voxel_features_sparse = x.features[batch_index == b]

            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0),
                                                                     -batch_dict['noise_rot'][b].unsqueeze(0))[
                    0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1

            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())

            voxels_2d_int = torch.Tensor(voxels_2d).to(x_rgb_batch.device).long()

            filter_idx = (0 <= voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0 <= voxels_2d_int[:, 0]) * (
                        voxels_2d_int[:, 0] < w)

            filter_idx_list.append(filter_idx)
            voxels_2d_int = voxels_2d_int[filter_idx]
            voxels_2d_int_list.append(voxels_2d_int)

            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], x_rgb_batch.shape[0]),
                                               device=x_rgb_batch.device)
            image_features_batch[filter_idx] = x_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            if fuse_sum:
                image_with_voxelfeature = image_features_batch + voxel_features_sparse
            else:
                image_with_voxelfeature = torch.cat([image_features_batch, voxel_features_sparse], dim=1)

            ############## Add ASPP

            image_with_voxelfeatures.append(image_with_voxelfeature)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures)
        return image_with_voxelfeatures

    def _gen_sparse_features(self, x, imps_3d, batch_dict, voxels_3d):
        """
            Generate the output sparse features from the focal sparse conv.
            Args:
                x: [N, C], lidar sparse features
                imps_3d: [N, kernelsize**3], the predicted importance values
                batch_dict: input and output information during forward
                voxels_3d: [N, 3], the 3d positions of voxel centers
        """
        batch_size = x.batch_size
        voxel_features_fore = []
        voxel_indices_fore = []
        voxel_features_back = []
        voxel_indices_back = []

        box_of_pts_cls_targets = []
        mask_voxels = []
        mask_kernel_list = []

        for b in range(batch_size):
            if self.training:
                index = x.indices[:, 0]
                batch_index = index == b
                mask_voxel = imps_3d[batch_index, -1].sigmoid()
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                mask_voxels.append(mask_voxel)
                gt_boxes = batch_dict['gt_boxes'][b, :, :-1].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                box_of_pts_cls_targets.append(box_of_pts_batch >= 0)

            features_fore, indices_fore, features_back, indices_back, mask_kernel = split_voxels(x, b, imps_3d,
                                                                                                 voxels_3d,
                                                                                                 self.kernel_offsets,
                                                                                                 mask_multi=self.mask_multi,
                                                                                                 topk=self.topk,
                                                                                                 threshold=self.threshold)

            mask_kernel_list.append(mask_kernel)
            voxel_features_fore.append(features_fore)
            voxel_indices_fore.append(indices_fore)
            voxel_features_back.append(features_back)
            voxel_indices_back.append(indices_back)

        voxel_features_fore = torch.cat(voxel_features_fore, dim=0)
        voxel_indices_fore = torch.cat(voxel_indices_fore, dim=0)
        voxel_features_back = torch.cat(voxel_features_back, dim=0)
        voxel_indices_back = torch.cat(voxel_indices_back, dim=0)
        mask_kernel = torch.cat(mask_kernel_list, dim=0)

        x_fore = spconv.SparseConvTensor(voxel_features_fore, voxel_indices_fore, x.spatial_shape, x.batch_size)
        x_back = spconv.SparseConvTensor(voxel_features_back, voxel_indices_back, x.spatial_shape, x.batch_size)

        loss_box_of_pts = 0
        if self.training:
            mask_voxels = torch.cat(mask_voxels)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            mask_voxels_two_classes = torch.cat([1 - mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return x_fore, x_back, loss_box_of_pts, mask_kernel

    def combine_out(self, x_fore, x_back, remove_repeat=False):
        """
            Combine the foreground and background sparse features together.
            Args:
                x_fore: [N1, C], foreground sparse features
                x_back: [N2, C], background sparse features
                remove_repeat: bool, whether to remove the spatial replicate features.
        """
        x_fore_features = torch.cat([x_fore.features, x_back.features], dim=0)
        x_fore_indices = torch.cat([x_fore.indices, x_back.indices], dim=0)

        if remove_repeat:
            index = x_fore_indices[:, 0]
            features_out_list = []
            indices_coords_out_list = []
            for b in range(x_fore.batch_size):
                batch_index = index == b
                features_out, indices_coords_out, _ = check_repeat(x_fore_features[batch_index],
                                                                   x_fore_indices[batch_index], flip_first=False)
                features_out_list.append(features_out)
                indices_coords_out_list.append(indices_coords_out)
            x_fore_features = torch.cat(features_out_list, dim=0)
            x_fore_indices = torch.cat(indices_coords_out_list, dim=0)

        x_fore = x_fore.replace_feature(x_fore_features)
        x_fore.indices = x_fore_indices

        return x_fore

    def forward(self, x, batch_dict, x_rgb=None):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]

        if self.use_img:
            features_multimodal = self.construct_multimodal_features(x, x_rgb, batch_dict)
            x_predict = spconv.SparseConvTensor(features_multimodal, x.indices, x.spatial_shape, x.batch_size)
        else:
            x_predict = self.conv_enlarge(x) if self.conv_enlarge else x

        imps_3d = self.conv_imp(x_predict).features

        x_fore, x_back, loss_box_of_pts, mask_kernel = self._gen_sparse_features(x, imps_3d, batch_dict, voxels_3d)

        if not self.skip_mask_kernel:
            x_fore = x_fore.replace_feature(x_fore.features * mask_kernel.unsqueeze(-1))
        out = self.combine_out(x_fore, x_back, remove_repeat=True)
        out = self.conv(out)

        if self.use_img:
            out = out.replace_feature(self.construct_multimodal_features(out, x_rgb, batch_dict, True))

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        return out, batch_dict, loss_box_of_pts
