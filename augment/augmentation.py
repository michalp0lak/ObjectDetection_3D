import numpy as np
import warnings
from ops.ops_numpy import bbox2corners3D, create_3D_rotations


class Augmentation():
    """Class consisting common augmentation methods for different pipelines."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def recenter(self, data, cfg):
        """Recenter pointcloud/features to origin.
        Typically used before rotating the pointcloud.
        Args:
            data: Pointcloud or features.
            cfg: config dict where
                Key 'dim' specifies dimension to be recentered.
        """
        if not cfg:
            return data
        dim = cfg.get('dim', [0, 1, 2])
        data[:, dim] = data[:, dim] - data.mean(0)[dim]
        return data

    def normalize(self, pc, feat, cfg):
        """Normalize pointcloud and/or features.
        Points are normalized in [0, 1] and features can take custom
        scale and bias.
        Args:
            pc: Pointcloud.
            feat: features.
            cfg: configuration dictionary.
        """
        if 'points' in cfg:
            cfg_p = cfg['points']
            if cfg_p.get('method', 'linear') == 'linear':
                pc -= pc.mean(0)
                pc /= (pc.max(0) - pc.min(0)).max()
            else:
                raise ValueError(f"Unsupported method : {cfg_p.get('method')}")

        if 'feat' in cfg and feat is not None:
            cfg_f = cfg['feat']
            if cfg_f.get('method', 'linear') == 'linear':
                bias = cfg_f.get('bias', 0)
                scale = cfg_f.get('scale', 1)
                feat -= bias
                feat /= scale
            else:
                raise ValueError(f"Unsupported method : {cfg_f.get('method')}")

        return pc, feat

    def rotate(self, pc, cfg):
        """Rotate the pointcloud.
        Two methods are supported. `vertical` rotates the pointcloud
        along yaw. `all` randomly rotates the pointcloud in all directions.
        Args:
            pc: Pointcloud to augment.
            cfg: configuration dictionary.
        """
        # Not checking for height dimension as preserving absolute height dimension improves some models.
        if np.abs(pc[:, :2].mean()) > 1e-2:
            warnings.warn(
                f"It is recommended to recenter the pointcloud before calling rotate."
            )

        method = cfg.get('method', 'vertical')

        if method == 'vertical':
            # Create random rotations
            theta = self.rng.random() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif method == 'all':

            # Choose two random angles for the first vector in polar coordinates
            theta = self.rng.random() * 2 * np.pi
            phi = (self.rng.random() - 0.5) * np.pi

            # Create the first vector in cartesian coordinates
            u = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])

            # Choose a random rotation angle
            alpha = self.rng.random() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)),
                                    np.reshape(alpha, (1, -1)))[0]
        else:
            raise ValueError(f"Unsupported method : {method}")

        R = R.astype(np.float32)

        return np.matmul(pc, R)

    def scale(self, pc, cfg):
        """Scale augmentation for pointcloud.
        If `scale_anisotropic` is True, each point is scaled differently.
        else, same scale from range ['min_s', 'max_s') is applied to each point.
        Args:
            pc: Pointcloud to scale.
            cfg: configuration dict.
        """
        # Choose random scales for each example
        scale_anisotropic = cfg.get('scale_anisotropic', False)
        min_s = cfg.get('min_s', 1.)
        max_s = cfg.get('max_s', 1.)

        if scale_anisotropic:
            scale = self.rng.random(pc.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = self.rng.random() * (max_s - min_s) + min_s

        return pc * scale

    def noise(self, pc, cfg):
        noise_std = cfg.get('noise_std', 0.001)
        noise = (self.rng.standard_normal(
            (pc.shape[0], pc.shape[1])) * noise_std).astype(np.float32)

        return pc + noise

    def augment(self, data):
        raise NotImplementedError(
            "Please use one of SemsegAugmentation or ObjdetAugmentation.")


class ObjdetAugmentation(Augmentation):
    """Class consisting different augmentation for Object Detection"""

    def __init__(self, cfg, seed=None):
        super(ObjdetAugmentation, self).__init__(cfg, seed=seed)

        # Raise warnings for misspelled/unimplemented methods.
        all_methods = [
            'recenter', 'normalize', 'rotate', 'scale', 'noise', 'PointShuffle',
            'ObjectRangeFilter', 'ObjectSample'
        ]
        for method in cfg:
            if method not in all_methods:
                warnings.warn(
                    f"Augmentation method : {method} does not exist. Please verify!"
                )

    def PointShuffle(self, data):
        """Shuffle Pointcloud."""
        self.rng.shuffle(data['point'])

        return data

    def in_range(self, pcd_range, box):

        return (box[:,0].min() > pcd_range[0]) & (box[:,0].max() < pcd_range[3]) & (box[:,1].min() > pcd_range[1]) & (
        box[:,1].max() < pcd_range[4]) & (box[:,2].min() > pcd_range[2]) & (box[:,2].max() < pcd_range[5])

    def ObjectRangeFilter(self, data, pcd_range):
        """Filter Objects in the given range."""
        pcd_range = np.array(pcd_range)

        filtered_boxes = []
        for box in data['bboxes']:
            if self.in_range(pcd_range, bbox2corners3D(box)):
                filtered_boxes.append(box)

        return {
            'point': data['point'],
            'labels': data['labels'],
            'bboxes': filtered_boxes,
        }

    def augment(self, data, attr, seed=None):
        """Augment object detection data.
        Available augmentations are:
            `ObjectSample`: Insert objects from ground truth database.
            `ObjectRangeFilter`: Filter pointcloud from given bounds.
            `PointShuffle`: Shuffle the pointcloud.
        Args:
            data: A dictionary object returned from the dataset class.
            attr: Attributes for current pointcloud.
        Returns:
            Augmented `data` dictionary.
        """
        cfg = self.cfg

        if cfg is None:
            return data

        # Override RNG for reproducibility with parallel dataloader.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if 'recenter' in cfg:
            if cfg['recenter']:
                data['point'] = self.recenter(data['point'], cfg['recenter'])

        if 'normalize' in cfg:
            data['point'], _ = self.normalize(data['point'], None,
                                              cfg['normalize'])

        if 'rotate' in cfg:
            data['point'] = self.rotate(data['point'], cfg['rotate'])

        if 'scale' in cfg:
            data['point'] = self.scale(data['point'], cfg['scale'])

        if cfg.get('ObjectRangeFilter', False):
            data = self.ObjectRangeFilter(data, cfg['ObjectRangeFilter']['point_cloud_range'])

        if cfg.get('PointShuffle', False):
            data = self.PointShuffle(data)

        return data