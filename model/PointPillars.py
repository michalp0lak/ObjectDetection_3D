import torch
from torch import nn
from torch.nn import functional as F

import spconv.pytorch as spconv

import numpy as np

from losses.cross_entropy import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.smooth_L1 import SmoothL1Loss

from augment.augmentation import ObjdetAugmentation

from model.base_model import BaseModel
from model.utils import Featurizer, CustomVoxelizer, ForegroundFilter, Anchor3DRangeGenerator, BBoxCoder, limit_period, multiclass_nms, get_paddings_indicator
from ops.ops_numba import VoxelGenerator
from ops.ops_numpy import global_outlier_check


class PointPillars(BaseModel):

    """Object localization model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Implementation based on
    https://github.com/isl-org/Open3D-ML/blob/f424215be133b8c2571e66bbab8fc5c4f2aaa931/ml3d/torch/models/point_pillars.py

    Args:
        name (string): Name of model.
            Default to "PointPillars".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        scatter: Config of PointPillarsScatter module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self,
                 name="PointPillars",
                 device="cuda",
                 classes=[],
                 input_features =[],
                 point_cloud_range=[],
                 preprocess = {},
                 augment={},
                 voxelize={},
                 voxel_encoder={},
                 vertical_encoder={},
                 backbone={},
                 neck={},
                 head={},
                 loss={},
                 **kwargs):

        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         device=device,
                         **kwargs)
        
        self.point_cloud_range = point_cloud_range
        
        self.classes = classes
        self.name2lbl = {n: i for i, n in enumerate(self.classes)}
        self.lbl2name = {i: n for i, n in enumerate(self.classes)}
        self.classes_ids = [i for i, _ in enumerate(self.classes)]
        self.input_features = input_features
        self.device = device

        self.custom_voxelizer = CustomVoxelizer(preprocess['voxelization'])
        self.featurizer = Featurizer(**preprocess['featurizer'])
        self.filter = ForegroundFilter(preprocess['filter'])

        self.augmentor = ObjdetAugmentation(augment, seed=self.rng)
        self.voxel_layer = PointPillarsVoxelization(point_cloud_range=self.point_cloud_range, device = self.device, **voxelize)
        self.voxel_encoder = PillarFeatureNet(point_cloud_range=self.point_cloud_range, **voxel_encoder)
        self.pseudoimage_generator = SparseMiddleExtractor(**vertical_encoder)

        self.backbone = BackboneDWS(**backbone)
        self.neck = BackboneUPS(**neck)
        self.sparse_rpn = SubmanifoldSparseRPN(**backbone)
        self.bbox_head = Anchor3DHead(num_classes=len(self.classes), **head)

        self.loss_cls = FocalLoss(**loss.get("focal", {}))
        self.loss_bbox = SmoothL1Loss(**loss.get("smooth_l1", {}))
        self.loss_dir = CrossEntropyLoss(**loss.get("cross_entropy", {}))

        self.to(device)

    def extract_feats(self, points):
        
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.pseudoimage_generator(voxel_features, coors, batch_size)
        #x = self.backbone(x)
        #x = self.neck(x)
        x = self.sparse_rpn(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        """
            Apply hard voxelization to points.

            Basically applies voxel_layer through batch, where each item in batch is individual point cloud
        """
        voxels, coors, num_points = [], [], []
        for individual_pc in points:

            individual_pc_voxels, individual_pc_coors, individual_pc_num_points = self.voxel_layer(individual_pc)

            voxels.append(individual_pc_voxels)
            coors.append(individual_pc_coors)
            num_points.append(individual_pc_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        
        # Here point cloud identifier is added. Pillars coordinates of each point cloud are 
        # concatenated with point cloud index 
        
        # (point cloud identifier in batch).
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward(self, inputs):
        inputs = inputs.point
        x = self.extract_feats(inputs)
        outs = self.bbox_head(x)
        return outs

    def get_optimizer(self, cfg):

        optimizer = torch.optim.AdamW(self.parameters(), **cfg)
        return optimizer

    def loss(self, results, inputs):

        scores, bboxes, dirs = results
        gt_bboxes = inputs.bboxes
        gt_labels = inputs.labels
        
        target_bboxes, target_idx, pos_idx, neg_idx = self.bbox_head.assign_bboxes(bboxes, gt_bboxes)
        avg_factor = pos_idx.size(0)

        # classification loss
        scores = scores.permute((0, 2, 3, 1)).reshape(-1, self.bbox_head.num_classes)

        target_labels = torch.full((scores.size(0),),
                                   self.bbox_head.num_classes,
                                   device=scores.device,
                                   dtype=gt_labels[0].dtype)

        target_labels[pos_idx] = torch.cat(gt_labels, axis=0)[target_idx]

        loss_cls = self.loss_cls(scores[torch.cat([pos_idx, neg_idx], axis=0)],
                                target_labels[torch.cat([pos_idx, neg_idx],
                                axis=0)],avg_factor=avg_factor)

        # remove invalid labels
        cond = (target_labels[pos_idx] >= 0) & (target_labels[pos_idx] < self.bbox_head.num_classes)
        pos_idx = pos_idx[cond]
        target_idx = target_idx[cond]
        target_bboxes = target_bboxes[cond]

        # generate and filter bboxes
        bboxes = bboxes.permute((0, 2, 3, 1)).reshape(-1, self.bbox_head.box_params_num)[pos_idx]
        dirs = dirs.permute((0, 2, 3, 1)).reshape(-1, 6)[pos_idx]
        dirs_x = dirs[:,:2]
        dirs_y = dirs[:,2:4]
        dirs_z = dirs[:,4:]

        if len(pos_idx) > 0:

            # direction classification loss
            # to discrete bins
            target_dirs_x = torch.cat(gt_bboxes, axis=0)[target_idx][:, -3]
            target_dirs_x = limit_period(target_dirs_x, 0, 2 * np.pi) # 180-360
            target_dirs_x = (target_dirs_x / np.pi).long() % 2

            target_dirs_y = torch.cat(gt_bboxes, axis=0)[target_idx][:, -2]
            target_dirs_y = limit_period(target_dirs_y, 0, 2 * np.pi) # 180-360
            target_dirs_y = (target_dirs_y / np.pi).long() % 2

            target_dirs_z = torch.cat(gt_bboxes, axis=0)[target_idx][:, -1]
            target_dirs_z = limit_period(target_dirs_z, 0, 2 * np.pi) # 180-360
            target_dirs_z = (target_dirs_z / np.pi).long() % 2


            loss_dir_x = self.loss_dir(dirs_x, target_dirs_x, avg_factor=avg_factor)
            loss_dir_y = self.loss_dir(dirs_y, target_dirs_y, avg_factor=avg_factor)
            loss_dir_z = self.loss_dir(dirs_z, target_dirs_z, avg_factor=avg_factor)

            # bbox loss
            # sinus difference transformation
            r0 = torch.sin(bboxes[:, -3:]) * torch.cos(target_bboxes[:, -3:])
            r1 = torch.cos(bboxes[:, -3:]) * torch.sin(target_bboxes[:, -3:])

            bboxes = torch.cat([bboxes[:, :-3], r0], axis=-1)
            target_bboxes = torch.cat([target_bboxes[:, :-3], r1], axis=-1)
            loss_bbox = self.loss_bbox(bboxes,target_bboxes, avg_factor=avg_factor)

        else:
            loss_cls = loss_cls.sum()
            loss_bbox = bboxes.sum()
            loss_dir_x = dirs_x.sum()
            loss_dir_y = dirs_y.sum()
            loss_dir_z = dirs_z.sum()

        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_dir_x': loss_dir_x,
            'loss_dir_y': loss_dir_y,
            'loss_dir_z': loss_dir_z
        }

    def preprocess(self, data, attr):

        # If num_workers > 0, use new RNG with unique seed for each thread.
        # Else, use default RNG.

        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng

        # Outlier check
        points = global_outlier_check(data['point'])

        #Filter points and bboxes in defined range
        points = np.array(points, dtype=np.float32)
        bboxes = np.array(data['bboxes'], dtype=np.float32)
        min_val = np.array(self.point_cloud_range[:3])
        max_val = np.array(self.point_cloud_range[3:])
 
        # Here points are filtered according to range
        points = points[np.where(
            np.all(np.logical_and(points[:, :3] >= min_val,points[:, :3] < max_val),axis=-1))]

        bboxes = bboxes[np.where(
            np.all(np.logical_and(bboxes[:, :2] >= min_val[:2], bboxes[:, :2] < max_val[:2]),axis=-1))]

        if points.shape[0] == 0:

            print('There are no points in defined range. Range is defined wrongly or this particular',
                                     'point cloud is affected with outliers: {}'.format(attr['name']))

        

        #points = self.custom_voxelizer.voxelize(points)

        points = points[:,self.input_features]

        data['point'] = points
        data['bboxes'] = bboxes

        #Augment data
        if attr['split'] not in ['test', 'testing', 'val', 'validation']:
            
            data = self.augmentor.augment(data, attr, seed=rng)

        new_data = {'point': data['point'], 'labels': data['labels'], 'bboxes': data['bboxes']}

        return new_data

    def transform(self, data, attr):

        #t_data = apply_some_transform(data['point'])
        t_data = data
        return t_data

    def inference_end(self, results):

        bboxes_b, scores_b, labels_b = self.bbox_head.get_bboxes(*results)

        inference_result = []
        for _bboxes, _scores, _labels in zip(bboxes_b,scores_b, labels_b):

            bboxes = _bboxes
            scores = _scores
            labels = _labels
            inference_result.append([])

            for bbox, score, label in zip(bboxes, scores, labels):
        
                inference_result[-1].append({'bbox': bbox, 'label': label, 'score': score})

        return inference_result

class PointPillarsVoxelization(torch.nn.Module):

    def __init__(self,
                 device,
                 voxel_size,
                 point_cloud_range,
                 max_voxel_points,
                 max_voxels):
         
        """Voxelization layer for the PointPillars model.
        Args:
            voxel_size: voxel edge lengths with format [x, y, z].
            point_cloud_range: The valid range of point coordinates as
                [x_min, y_min, z_min, x_max, y_max, z_max].
            max_num_points: The maximum number of points per voxel.
            max_voxels: The maximum number of voxels. May be a tuple with
                values for training and testing.
        """
        super().__init__()
        
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = np.array(voxel_size)
        self.max_voxel_points = max_voxel_points
        self.max_voxels = max_voxels
        self.device = device

    def forward(self, points):
        """Forward function.
        Args:
            points_feats: Tensor with point coordinates and features. The shape
                is [N, 3+C] with N as the number of points and C as the number
                of feature channels.
        Returns:
            (out_voxels, out_coords, out_num_points).
            * out_voxels is a dense list of point coordinates and features for
              each voxel. The shape is [num_voxels, max_num_points, 3+C].
            * out_coords is tensor with the integer voxel coords and shape
              [num_voxels,3]. Note that the order of dims is [z,y,x].
            * out_num_points is a 1D tensor with the number of points for each
              voxel.
        """

        VG = VoxelGenerator(self.voxel_size, self.point_cloud_range, self.max_voxel_points, self.max_voxels)

        voxels, coords, num_points = VG.generate(points, self.max_voxels, self.point_cloud_range, True)

        out_voxels = torch.tensor(voxels, dtype=torch.float32).to(self.device)
        out_coords = torch.tensor(coords[:, [2, 1, 0]], dtype=torch.int64).to(self.device)
        out_num_points = torch.tensor(num_points, dtype=torch.int64).to(self.device)

        return out_voxels, out_coords, out_num_points


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.
    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self, in_channels, out_channels, last_layer=False, mode='avg'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer

        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    #@auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxel_points=None, aligned_distance=None):

        """Forward function.
        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.
        Returns:
            torch.Tensor: Features of Pillars.
        """

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0,2,1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]

        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(dim=1,
                          keepdim=True) / num_voxel_points.type_as(inputs).view(-1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.
    The network prepares the pillar features and performs forward pass
    through PFNLayers.
    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size.
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 voxel_size,
                 point_cloud_range):

        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0

        in_channels += 5

        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []

        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
                out_filters -= 1

            pfn_layers.append(
                PFNLayer(in_filters,
                         out_filters,
                         last_layer=last_layer,
                         mode='max'))

        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.fp16_enabled = False
        self.point_cloud_range = point_cloud_range
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + self.point_cloud_range[0]
        self.y_offset = self.vy / 2 + self.point_cloud_range[1]

    def forward(self, features, num_points, coors):
        """Forward function.
        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.
        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]

        # Find distance of x, y, and z from cluster center
        voxels_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)

        voxels_centroid_dist = features[:, :, :3] - voxels_mean
        features_ls.append(voxels_centroid_dist)

        # Find distance of x, y, and z from pillar center
        pillar_base_center = features[:, :, :2].clone().detach()

        pillar_base_center[:, :, 0] = pillar_base_center[:, :, 0] - (
            coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
            self.x_offset)

        pillar_base_center[:, :, 1] = pillar_base_center[:, :, 1] - (
            coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
            self.y_offset)

        features_ls.append(pillar_base_center)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that empty pillars remain set to zeros.

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return torch.cat((features.squeeze(1), num_points.view(-1,1)), dim = -1)


class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 in_channels,
                 out_channels
                ):

        super(SparseMiddleExtractor, self).__init__()

        self.sparse_shape = output_shape
        middle_layers = []

        num_filters = [in_channels] + out_channels
        filters_pairs_d = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for index, pair in enumerate(filters_pairs_d):

            i = pair[0]
            o = pair[1]

            middle_layers.append(
                spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm{}".format(index)))
            middle_layers.append(nn.BatchNorm1d(o))
            middle_layers.append(nn.ReLU())

            middle_layers.append(
                spconv.SparseConv3d(o,o, (3, 1, 1), (2, 1, 1),bias=False))               
            middle_layers.append(nn.BatchNorm1d(o))
            middle_layers.append(nn.ReLU())

        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)

        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret

class BackboneDWS(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.
    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):

        super(BackboneDWS, self).__init__()

        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []

        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_filters[i],
                          out_channels[i],
                          3,
                          bias=False,
                          stride=layer_strides[i],
                          padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(out_channels[i],
                              out_channels[i],
                              3,
                              bias=False,
                              padding=1))
                block.append(
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).
        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class BackboneUPS(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.
    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False):

        super(BackboneUPS, self).__init__()
        
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(in_channels=in_channels[i],
                                           out_channels=out_channel,
                                           kernel_size=stride,
                                           stride=stride,
                                           bias=False)

            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            deblocks.append(deblock)

        self.deblocks = nn.ModuleList(deblocks)
        self.init_weights()

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.
        Returns:
            torch.Tensor: Feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out


class SubmanifoldSparseRPN(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.
    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=[512, 1024, 2048],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]       
                ):

        super(SubmanifoldSparseRPN, self).__init__()

        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []

        for i, layer_num in enumerate(layer_nums):
            
            blocks.append(spconv.SubMConv2d(in_filters[i], out_channels[i], 3, layer_strides[i], padding = 1, bias=False))
            blocks.append(nn.BatchNorm1d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))
            
            for j in range(layer_num):
                blocks.append(
                    spconv.SubMConv2d(out_channels[i], out_channels[i], 3, layer_strides[i], padding = 1, bias=False)
                    #spconv.SparseConv2d(out_channels[i], out_channels[i], 3, 1, bias=False, padding=1)
                    )
                blocks.append(
                    nn.BatchNorm1d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

        self.blocks = spconv.SparseSequential(*blocks)


    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).
        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        
        values_batch = []
        coords_batch = []

        for i in range(x.shape[0]):

            # Transform each item (point cloud) in batch to sparse tensor

            # In each batch item find not empty voxels
            indexes = torch.where((x[i,:,:,:] != 0).any(axis=0))
            # Concat tensor indexes
            coord = torch.cat((indexes[0].unsqueeze(1), indexes[1].unsqueeze(1)), axis=1)
            # Add batch index
            coord = F.pad(coord, (1, 0), mode='constant', value=i)
            coord = coord.int()
            # Select pointn cloud not-empty voxels features
            values = x[i,:,indexes[0],indexes[1]].T

            coords_batch.append(coord)
            values_batch.append(values)    

        values_batch = torch.cat(values_batch, dim=0)
        coords_batch = torch.cat(coords_batch, dim=0)
        
        x = spconv.SparseConvTensor(values_batch, coords_batch, x.shape[-2:], x.shape[0])
        
        x = self.blocks(x).dense()

        return x

class Anchor3DHead(nn.Module):

    def __init__(self,
                 num_classes=1,
                 in_channels=384,
                 nms_dim = 2,
                 nms_pre=100,
                 nms_thresh = 0.7,
                 score_thr=0.1,
                 box_params_num = 9,
                 dir_offset=0,
                 ranges=[],
                 sizes=[],
                 rotations=[],
                 iou_thr=[]):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.nms_pre = nms_pre
        self.nms_thresh = nms_thresh
        self.score_thr = score_thr
        self.dir_offset = dir_offset
        self.iou_thr = iou_thr
        self.sizes = sizes
        self.ranges = ranges
        self.rotations = rotations
        self.box_params_num = box_params_num
        self.nms_dim = nms_dim

        if len(self.iou_thr) != num_classes:
            assert len(self.iou_thr) == 1
            self.iou_thr = self.iou_thr * num_classes
        assert len(self.iou_thr) == num_classes

        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(ranges=self.ranges,
                                                       sizes=self.sizes,
                                                       rotations=self.rotations,
                                                       box_params_num = self.box_params_num)

        self.num_anchors = self.anchor_generator.num_base_anchors

        # build box coder
        self.bbox_coder = BBoxCoder()
        self.fp16_enabled = False

        #Initialize neural network layers of the head.
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.in_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels,self.num_anchors * self.box_params_num, 1)
        self.conv_dir_cls_x = nn.Conv2d(self.in_channels, self.num_anchors * 2, 1)
        self.conv_dir_cls_y = nn.Conv2d(self.in_channels, self.num_anchors * 2, 1)
        self.conv_dir_cls_z = nn.Conv2d(self.in_channels, self.num_anchors * 2, 1)

        self.init_weights()

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """Initialize conv/fc bias value according to giving probablity."""
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))

        return bias_init

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = self.bias_init_with_prob(0.01)
        self.normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        self.normal_init(self.conv_reg, std=0.01)

    def forward(self, x):
        """Forward function on a feature map.
        Args:
            x (torch.Tensor): Input features.
        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds_x = self.conv_dir_cls_x(x)
        dir_cls_preds_y = self.conv_dir_cls_y(x)
        dir_cls_preds_z = self.conv_dir_cls_z(x)

        return cls_score, bbox_pred, torch.cat([dir_cls_preds_x, dir_cls_preds_y, dir_cls_preds_z], axis=1)

    def assign_bboxes(self, pred_bboxes, target_bboxes):
        """Assigns target bboxes to given anchors.
        Args:
            pred_bboxes (torch.Tensor): Bbox predictions (anchors).
            target_bboxes (torch.Tensor): Bbox targets.
        Returns:
            torch.Tensor: Assigned target bboxes for each given anchor.
            torch.Tensor: Flat index of matched targets.
            torch.Tensor: Index of positive matches.
            torch.Tensor: Index of negative matches.
        """
        # compute all anchors for each point cloud in batch

        if self.nms_dim == 3:
            from ops.ops_torch import bbox2corners3D as box2vertices
            from ops.ops_torch import box3d_overlap as box_overlap

        elif self.nms_dim == 2:
            from ops.ops_torch import bbox2rotated_corners2D as box2vertices
            from ops.ops_torch import bbox_iou2D as box_overlap

        anchors = [
            self.anchor_generator.grid_anchors(pred_bboxes.shape[-2:],
                                               device=pred_bboxes.device)
            for _ in range(len(target_bboxes))
        ]
        
        # compute size of anchors for each given class
        anchors_cnt = torch.tensor(anchors[0].shape[:-1]).prod()
        rot_angles = anchors[0].shape[-2]

        # init the tensors for the final result
        assigned_bboxes, target_idxs, pos_idxs, neg_idxs = [], [], [], []

        def flatten_idx(idx, j):

            """Inject class dimension in the given indices (...
            z * rot_angles + x) --> (.. z * num_classes * rot_angles + j * rot_angles + x)
            """

            z = torch.div(idx, rot_angles, rounding_mode='trunc')
            x = idx % rot_angles

            return z * self.num_classes * rot_angles + j * rot_angles + x

        idx_off = 0
   
        # For each point cloud in batch: i -> index of item/point cloud in batch
        for i in range(len(target_bboxes)):
            
            # j -> class index
            # I have single class. And in each pseudopixel I want several base anchors given by combinations of sizes and rotations.
            # I suppose tree (single class) anchors of different sizes and rotations.
            for j in range(self.num_classes):

                # Use all anchors as 9-element vectors
                anchors_stride = anchors[i].reshape(-1, self.box_params_num)
                pred_bboxes = pred_bboxes.reshape(-1, self.box_params_num)

                # If there is not any ground truth bounding box
                if target_bboxes[i].shape[0] == 0:
                    assigned_bboxes.append(
                        torch.zeros((0, self.box_params_num), device=pred_bboxes.device))
                    target_idxs.append(
                        torch.zeros((0,),
                                    dtype=torch.long,
                                    device=pred_bboxes.device))
                    pos_idxs.append(
                        torch.zeros((0,),
                                    dtype=torch.long,
                                    device=pred_bboxes.device))
                    neg_idxs.append(
                        torch.zeros((0,),
                                    dtype=torch.long,
                                    device=pred_bboxes.device))
                    continue
                
                # compute a 3D IoU
                overlaps = box_overlap(box2vertices(target_bboxes[i]), 
                                      box2vertices(anchors_stride))
                
                # for each anchor the gt with max IoU
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)
                
                # for each gt the anchor with max IoU
                gt_max_overlaps, gt_max_indxs = overlaps.max(dim=1)
                pos_idx = max_overlaps >= self.iou_thr[j][1]
                neg_idx = (max_overlaps >= 0) & (max_overlaps < self.iou_thr[j][0])

                # low-quality matching
                for k in range(len(target_bboxes[i])):
                    if gt_max_overlaps[k] >= self.iou_thr[j][0]:
                        pos_idx[overlaps[k, :] == gt_max_overlaps[k]] = True
                
                assigned_bboxes.append(self.bbox_coder.encode(anchors_stride[pos_idx],
                                       target_bboxes[i][argmax_overlaps[pos_idx]]))

                target_idxs.append(argmax_overlaps[pos_idx] + idx_off)

                # store global indices in list
                pos_idx = flatten_idx(
                    pos_idx.nonzero(as_tuple=False).squeeze(-1),
                    j) + i * anchors_cnt
                neg_idx = flatten_idx(
                    neg_idx.nonzero(as_tuple=False).squeeze(-1),
                    j) + i * anchors_cnt

                pos_idxs.append(pos_idx)
                neg_idxs.append(neg_idx)

            # compute offset for index computation
            idx_off += len(target_bboxes[i])

        return (torch.cat(assigned_bboxes,axis=0), torch.cat(target_idxs, axis=0),
                torch.cat(pos_idxs, axis=0), torch.cat(neg_idxs, axis=0))

    def get_bboxes(self, cls_scores, bbox_preds, dir_preds):

        """Get bboxes of anchor head.
        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.
        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """

        bboxes, scores, labels = [], [], []
        for cls_score, bbox_pred, dir_pred in zip(cls_scores, bbox_preds,dir_preds):
            b, s, l = self.get_bboxes_single(cls_score, bbox_pred, dir_pred)

            bboxes.append(b)
            scores.append(s)
            labels.append(l)

        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_preds):

        """Get bboxes of anchor head.
        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.
        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        assert cls_scores.size()[-2:] == dir_preds.size()[-2:]

        # Get anchors
        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:],device=cls_scores.device)
        anchors = anchors.reshape(-1, self.box_params_num)

        # Reshape result
        dir_preds = dir_preds.permute(1, 2, 0).reshape(-1, 6)
        dir_scores_x = torch.max(dir_preds[:,:2], dim=-1)[1]
        dir_scores_y = torch.max(dir_preds[:,2:4], dim=-1)[1]
        dir_scores_z = torch.max(dir_preds[:,4:], dim=-1)[1]

        cls_scores = cls_scores.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_scores.sigmoid()

        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, self.box_params_num)
        bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        if scores.shape[0] > self.nms_pre:

            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(self.nms_pre)
            anchors = anchors[topk_inds, :]
            bboxes = bboxes[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_scores_x = dir_scores_x[topk_inds]
            dir_scores_y = dir_scores_y[topk_inds]
            dir_scores_z = dir_scores_z[topk_inds]
        
        idxs = multiclass_nms(bboxes, scores, self.score_thr, self.nms_thresh, self.nms_dim)

        labels = [
            torch.full((len(idxs[i]),), i, dtype=torch.long)
            for i in range(self.num_classes)
        ]

        labels = torch.cat(labels)
        scores = [scores[idxs[i], i] for i in range(self.num_classes)]
        scores = torch.cat(scores)

        idxs = torch.cat(idxs)

        bboxes = bboxes[idxs]
        dir_scores_x = dir_scores_x[idxs]
        dir_scores_y = dir_scores_y[idxs]
        dir_scores_z = dir_scores_z[idxs]
        
        if bboxes.shape[0] > 0:

            dir_rot_x = limit_period(bboxes[..., -3] - self.dir_offset, 1, np.pi)
            bboxes[..., -3] = (dir_rot_x + self.dir_offset + np.pi * dir_scores_x.to(bboxes.dtype))
            dir_rot_y = limit_period(bboxes[..., -2] - self.dir_offset, 1, np.pi)
            bboxes[..., -2] = (dir_rot_y + self.dir_offset + np.pi * dir_scores_y.to(bboxes.dtype))
            dir_rot_z = limit_period(bboxes[..., -1] - self.dir_offset, 1, np.pi)
            bboxes[..., -1] = (dir_rot_z + self.dir_offset + np.pi * dir_scores_z.to(bboxes.dtype))

        return bboxes, scores, labels