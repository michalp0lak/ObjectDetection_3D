import torch
import numpy as np
import open3d as o3d
from ops.ops_numba import CustomVoxelGenerator
import xgboost as xgb
from model.mlp import MLP

class CustomVoxelizer():

    def __init__(self, cfg):

        self.cfg = cfg
        self.VG = CustomVoxelGenerator(**self.cfg)

    def voxelize(self, point_cloud):

        point_cloud_range = point_cloud[:,:3].min(axis=0).tolist() + point_cloud[:,:3].max(axis=0).tolist()
        range_dims = point_cloud[:,:3].max(axis=0) - point_cloud[:,:3].min(axis=0)
        density = point_cloud.shape[0]/np.prod(range_dims)

        a = 20000
        b = 0.01
        c = 70000
        voxel_limit = 3000000

        if density > 10:

            max_voxels = np.min([int(a*np.exp(b*density) + c),point_cloud.shape[0]])

            if max_voxels <= point_cloud.shape[0]:

                max_voxels = np.min([max_voxels, voxel_limit])

                vox, co, vp = self.VG.generate(point_cloud, point_cloud_range, max_voxels)
                point_cloud = np.sum(vox, axis=1)/vp.reshape(-1,1)
        else:

            if point_cloud.shape[0] > voxel_limit:

                vox, co, vp = self.VG.generate(point_cloud, point_cloud_range, voxel_limit)
                point_cloud = np.sum(vox, axis=1)/vp.reshape(-1,1)

        return np.concatenate((point_cloud, vp.reshape(-1,1)), axis=1)


class Featurizer():

    """
        Class evaluates normals and FPFH (fast point feature histograms) for point cloud and returns dict of two items.

        Args:
            normal_rad: radius defining ball neighborhood of point for normals evaluation
            normal_max_nn: maximal number of nearest neighbors in ball neighborhood for normals evaluation
            fpfh_rad: radius defining ball neighborhood of point for fpfh evaluation
            fpfh_max_nn: maximal number of nearest neighbors in ball neighborhood for fpfh evaluation

        Returns:
            Dictionary with two items/matrices -> normals and fpfh
    """

    def __init__(self, normal_rad: float, normal_max_nn: int, fpfh_rad: float, fpfh_max_nn: int, **kwargs):

        assert (type(normal_rad) == float), 'Radius for normals evaluation has be float value'
        assert (type(normal_max_nn) == int), 'Maximum value of nearest neighbors for normals evaluation has be integer value'
        assert (type(fpfh_rad) == float), 'Radius for fpfhs evaluation has be float value'
        assert (type(fpfh_max_nn) == int), 'Maximum value of nearest neighbors for fpfhs evaluation has be integer value'

        self.normal_rad = normal_rad
        self.normal_max_nn = normal_max_nn
        self.fpfh_rad = fpfh_rad
        self.fpfh_max_nn = fpfh_max_nn

    def generate_features(self, point_cloud):

        assert (type(point_cloud) == np.ndarray) and (point_cloud.shape[0] > 0), 'Input matrix has to be numpy array and has not to be empty'

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_rad, max_nn=self.normal_max_nn))
        normals = np.asarray(pcd.normals)[:, :]

        # This time-heavy operation (for 15000000 points around 60s)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_rad, 
                                                               max_nn=self.fpfh_max_nn))

        ## Concatenate all evaluated features with original points
        point_cloud = np.concatenate((np.concatenate((point_cloud, normals), axis = 1),
                                      fpfh.data.T), axis = 1)

        return point_cloud

class ForegroundFilter():

    def __init__(self, cfg):

        self.path = cfg.get('path', None)
        self.filter_type = cfg.get('filter_type', None)
        assert self.filter_type in ['xgboost', 'mlp'], 'Filter segmenting trunk points has to be xgboost or mlp'
        self.filter_cfg = cfg[self.filter_type]
        self.trunk_prob = self.filter_cfg.get('trunk_confidence')
        assert (type(self.trunk_prob)==float) and ((self.trunk_prob > 0) and (self.trunk_prob < 1)), (
            'Threshold value is not valid, has to be in (0,1) interval')
        self.init_model()

    def init_model(self):

        if self.filter_type == 'xgboost':

            self.foreground_filter = xgb.Booster()
            self.foreground_filter.load_model('{}{}'.format(self.path, self.filter_type))
        
        elif self.filter_type == 'mlp':

            self.foreground_filter = MLP(**self.filter_cfg)
            ckpt = torch.load('{}{}.pth'.format(self.path,self.filter_type))
            self.foreground_filter.load_state_dict(ckpt['model_state_dict'])
        
    def evaluate(self, X):

        if self.filter_type == 'xgboost':

            X = xgb.DMatrix(X)
            prediction = self.foreground_filter.predict(X)
            return (prediction > self.trunk_prob).astype(np.uint8)

        elif self.filter_type == 'mlp':

            X = torch.from_numpy(X).to(self.filter_cfg['device'])
            X = X.unsqueeze(0)
            X = X.type(torch.float32)
            prediction = self.foreground_filter(X)

            return (prediction > self.trunk_prob).squeeze(0).squeeze(-1).cpu().detach().numpy()
            

class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.
    This anchor generator generates anchors by the given range in different
    feature levels.
    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        rotations (list[float]): Rotations of anchors in a feature grid.
    """

    def __init__(self,
                 ranges,
                 sizes,
                 rotations,
                 box_params_num):

        self.sizes = sizes
        self.ranges = ranges
        self.rotations = rotations
        self.box_params_num = box_params_num

    @property
    def num_base_anchors(self):

        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)

        return num_rot * num_size

    def grid_anchors(self, featmap_size, device='cuda'):
        """Generate grid anchors of a single level feature map.
        This function is usually called by method ``self.grid_anchors``.
        Args:
            featmap_size (tuple[int]): Size of the feature map.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        if len(featmap_size) == 2:
            featmap_size = [1, featmap_size[0], featmap_size[1]]

        mr_anchors = []
        for anchor_range in self.ranges:
            for anchor_size in self.sizes:
                for anchor_rotation in self.rotations:

                    mr_anchors.append(
                        self.anchors_single_range(featmap_size,
                                                anchor_range,
                                                anchor_size,
                                                anchor_rotation,
                                                device=device))

        mr_anchors = torch.cat(mr_anchors, dim=-3).reshape(featmap_size[0],featmap_size[1],featmap_size[2],
                                                            len(self.sizes), len(self.rotations), self.box_params_num)
        return mr_anchors

     
    def anchors_single_range(self,
                                feature_size,
                                anchor_range,
                                sizes,
                                rotations,
                                device='cuda'):

        """Generate anchors in a single range.
        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            sizes (list[list] | np.ndarray | torch.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | torch.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.
        Returns:
            torch.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, box_params_num].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]

        
        anchor_range = torch.tensor(anchor_range, device=device)
        
        z_centers = torch.linspace(anchor_range[2],
                                    anchor_range[5],
                                    feature_size[0],
                                    device=device)

        y_centers = torch.linspace(anchor_range[1],
                                    anchor_range[4],
                                    feature_size[1],
                                    device=device)
        x_centers = torch.linspace(anchor_range[0],
                                    anchor_range[3],
                                    feature_size[2],
                                    device=device)

        sizes = torch.tensor(sizes, device=device).reshape(-1, 3)
        rotations = torch.tensor(rotations, device=device)

        rets = torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

        rets = list(rets)
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, 1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        rotations = rotations.reshape([1, 1, 1, 1, 1, 3])
        tile_rotation_shape = list(rets[0].shape)
        tile_rotation_shape[4] = 1
        rotations = rotations.repeat(tile_rotation_shape)
        rets.insert(4, rotations)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        return ret

class BBoxCoder(object):
    """Bbox Coder for 3D boxes.
    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self):
        super(BBoxCoder, self).__init__()

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, ddx, ddy, ddz, drx, dry, drz)
           that can be used to transform the `src_boxes` into the
        `target_boxes`.
        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.
        Returns:
            torch.Tensor: Box transformation deltas.
        """
        xa, ya, za, dxa, dya, dza, rxa, rya, rza = torch.split(src_boxes, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rxg, ryg, rzg = torch.split(dst_boxes, 1, dim=-1)

        zg = zg + dzg/2
        za = za + dza / 2
        diagonal = torch.sqrt(dxa**2 + dya**2)

        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza

        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)

        rxt = rxg - rxa
        ryt = ryg - rya
        rzt = rzg - rza

        return torch.cat([xt, yt, zt, dxt, dyt, dzt, rxt, ryt, rzt], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, ddx, ddy, ddz, drx, dry, drz) to
        `boxes`.
        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 9).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 9) [x, y, z, dx, dy, dz, rx, ry, rz].
        Returns:
            torch.Tensor: Decoded boxes.
        """
        xa, ya, za, dxa, dya, dza, rxa, rya, rza = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rxt, ryt, rzt = torch.split(deltas, 1, dim=-1)

        za = za + dza / 2
        diagonal = torch.sqrt(dxa**2 + dya**2)

        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        rxg = rxt + rxa
        ryg = ryt + rya
        rzg = rzt + rza
        
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rxg, ryg, rzg], dim=-1)

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.
    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def multiclass_nms(boxes, scores, score_thr, iou_thr, nms_dim):
    """Multi-class nms for 3D boxes.
    Args:
        boxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        scores (torch.Tensor): Multi-level boxes with shape
            (N, ). N is the number of boxes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.
        iou_thr (float): IoU threshold to surpress highly overlapping bounding boxes.
    Returns:
        list[torch.Tensor]: Return a list of indices after nms,
            with an entry for each class.
    """

    if nms_dim == 3:
        from ops.ops_torch import bbox2corners3D as box2vertices
        from ops.ops_torch import box3d_overlap as box_overlap

    elif nms_dim == 2:
        from ops.ops_torch import bbox2rotated_corners2D as box2vertices
        from ops.ops_torch import bbox_iou2D as box_overlap

    idxs = []

    # For each class
    for i in range(scores.shape[1]):
        # Check for all datums if predicted score is bigger than threshold
        cls_inds = scores[:, i] > score_thr

        # If class was not predicted strongly enough for any datum
        if not cls_inds.any():
            
            # Append empty tensor and go to another class
            idxs.append(torch.tensor([], dtype=torch.long, device=cls_inds.device))
            continue

        # Tensor of original indices of datums, which were selected for given class
        orig_idx = torch.arange(cls_inds.shape[0], device=cls_inds.device,dtype=torch.long)[cls_inds]

        # Sample valid scores and boxes
        _scores = scores[cls_inds, i]
        _boxes = boxes[cls_inds, :]

        box_vertices = box2vertices(_boxes)
        scores_sorted = torch.argsort(_scores, dim=0, descending=True)

        orig_idx = orig_idx[scores_sorted]
        boxes_sorted = box_vertices[scores_sorted,:]

        box_indices = torch.arange(0, boxes_sorted.shape[0]).cuda()
        suppressed_box_indices = []

        while box_indices.shape[0] > 0:
            # If box with highest classification score is not among suppressed bounding boxes
            if box_indices[0] not in suppressed_box_indices:
                # Choose box with best score
                selected_box = box_indices[0]          

                selected_iou = box_overlap(boxes_sorted[box_indices], boxes_sorted[selected_box].unsqueeze(0))
                mask_iou = (selected_iou > iou_thr).squeeze(-1)

                mask_indices = box_indices != selected_box
                mask = mask_iou & mask_indices
                suppressed_box_indices.append(box_indices[mask].tolist())
            
            box_indices = box_indices[torch.logical_not(mask)]
            box_indices = box_indices[1:]
        
        suppressed_box_indices = [idx for slist in suppressed_box_indices for idx in slist]
        preserved_box_indexes = list(set(np.arange(0, boxes_sorted.shape[0]).tolist()) - set(suppressed_box_indices))
        idxs.append(orig_idx[preserved_box_indexes])

    return idxs

def get_paddings_indicator_np(actual_num, max_num):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel
    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = actual_num.reshape(-1, 1)
    max_num = np.arange(max_num).reshape(1,-1)
    paddings_indicator = actual_num > max_num
    
    return paddings_indicator

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel
    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int,
                           device=actual_num.device).view(max_num_shape)

    paddings_indicator = actual_num.int() > max_num
    
    return paddings_indicator
