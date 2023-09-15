from abc import ABC, abstractmethod
import logging
import numpy as np

# Import package modules
from config import Config

log = logging.getLogger(__name__)

class BaseDataset(ABC):
    """The base dataset class that is used by all other datasets.
    All datasets must inherit from this class and implement the functions in order to be
    compatible with pipelines.
    Args:
        **kwargs: The configuration of the model as keyword arguments.
    Attributes:
        cfg: The configuration file as Config object that stores the keyword
            arguments that were passed to the constructor.
        name: The name of the dataset.
    """

    def __init__(self, **kwargs):
        
        """Initialize the class by passing the dataset path."""
        if kwargs['dataset_path'] is None:
            raise KeyError("Provide dataset_path to initialize the dataset")

        if kwargs['name'] is None:
            raise KeyError("Provide dataset name to initialize it")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    @staticmethod
    @abstractmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.
        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """

    @abstractmethod
    def get_split(self, split):
        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return


class BaseDatasetSplit(ABC):
    """The base class for dataset splits.
    This class provides access to the data of a specified subset or split of a dataset.
    Args:
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.
    Attributes:
        cfg: Shortcut to the config of the dataset object.
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.
    """

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    @abstractmethod
    def __len__(self):
        """Returns the number of samples in the split."""
        return 0

    @abstractmethod
    def get_data(self, idx):
        """Returns the data for the given index."""
        return {}

    @abstractmethod
    def get_attr(self, idx):
        """Returns the attributes for the given index."""
        return {}