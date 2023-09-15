import numpy as np
import glob
from pathlib import Path
import logging

from dataset.base_dataset import BaseDataset, BaseDatasetSplit

log = logging.getLogger(__name__)


# Expect point clouds to be in npy format with train, val and test files in separate folders.
# Expected format of npy files : ['x', 'y', 'z', 'feat_1', 'feat_2', ........,'feat_n'].


class ForestSplit(BaseDatasetSplit):
    """This class is used to create a custom dataset split.
    Initialize the class.
    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
        'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.
    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)

        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset
        self.selected_features = self.cfg.get('input_features', None)

    def __len__(self):
        return len(self.path_list)

    def read_lidar(self,path):
        """Reads lidar data from the path provided.
        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return np.load(path)

    def get_data(self, idx):

        pc_path = self.path_list[idx]
        bbox_path = self.path_list[idx].split('.')[0] + '_bbx.npy'

        data = self.read_lidar(pc_path)

        bboxes = np.load(bbox_path)
        bboxes[:,2] = bboxes[:,2] - bboxes[:,5]/2
        bboxes[:,6:] = np.deg2rad(bboxes[:,6:])
        
        points = np.array(data, dtype=np.float32)

        # Get translation vector to shift PC and boxes to origin
        shift = np.min(points[:,:3], axis=0)

        points[:,:3] = points[:,:3] - shift
        bboxes[:,:3] = bboxes[:,:3] - shift

        out_of_scene = (bboxes[:,0]<0) | (bboxes[:,1]<0)  
        if sum(out_of_scene): print("Boxes out of scene {}".format(bboxes[out_of_scene]))

        label = np.zeros(shape=(bboxes.shape[0],))

        return {'point': points, 'labels': label, 'bboxes': bboxes}

    def get_attr(self, idx):

        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.npy', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}

        return attr
    
class Forest3D(BaseDataset):
    """A template for customized dataset that you can use with a dataloader to
    feed data when training a model. This inherits all functions from the base
    dataset and can be modified by users. Initialize the function by passing the
    dataset and other details.
    Args:
        dataset_path: The path to the dataset to use.
        name: The name of the dataset.
        cache_dir: The directory where the cache is stored.
        use_cache: Indicates if the dataset should be cached.
        num_points: The maximum number of points to use when splitting the dataset.
        ignored_label_inds: A list of labels that should be ignored in the dataset.
        test_result_folder: The folder where the test results should be stored.
    """

    def __init__(self,
                 dataset_path,
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         **kwargs)

        cfg = self.cfg

        self.dataset_path = cfg.dataset_path 

        self.train_dir = str(Path(cfg.dataset_path) / 'training')
        self.val_dir = str(Path(cfg.dataset_path) / 'validation')
        self.test_dir = str(Path(cfg.dataset_path) / 'testing')

        self.train_files = [f for f in glob.glob(self.train_dir + "/*.npy") if not 'bbx' in f]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*.npy") if not 'bbx' in f]
        self.test_files = [f for f in glob.glob(self.test_dir + "/*.npy") if not 'bbx' in f]

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.
        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'Tree'
        }

        return label_to_names

    def get_split(self, split):

        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
        """

        return ForestSplit(self, split=split)

    def get_split_list(self, split):

        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'testing', 'validation'.
        Returns:
            A dataset split object providing the requested subset of the data.
        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'test', 'validation'.
        """

        if split in ['test', 'testing']:
            self.rng.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            self.rng.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            self.rng.shuffle(self.train_files)
            return self.train_files
        else:
            raise ValueError("Invalid split {}".format(split))