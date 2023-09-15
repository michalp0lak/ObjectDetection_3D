from torch.utils.data import Dataset
import torch
import numpy as np

class ObjectDetectBatcher(object):

    def __init__(self, batch, box_params_num):
        """Initialize.
        Args:
            batches: A batch of data
        Returns:
            class: The corresponding class.
        """
        self.point = []
        self.labels = []
        self.bboxes = []
        self.attr = []
        self.box_params_num = box_params_num

        for batch_item in batch:

            self.attr.append(batch_item['attr'])
            
            data = batch_item['data']
            
            self.point.append(data['point'].astype(np.float32))
            self.labels.append(data['labels'].astype(np.int64) if 'labels' in data else None)
            
            if len(data.get('bboxes', [])) > 0:
                self.bboxes.append(data['bboxes'].astype(np.float32) if 'bboxes' in data else None)
            else:
                self.bboxes.append(np.zeros((0, self.box_params_num)))

    def pin_memory(self):
        for i in range(len(self.point)):
            self.point[i] = self.point[i].pin_memory()
            if self.labels[i] is not None:
                self.labels[i] = self.labels[i].pin_memory()
            if self.bboxes[i] is not None:
                self.bboxes[i] = self.bboxes[i].pin_memory()

        return self

    def to(self, device):

        for i in range(len(self.point)):

            #self.point[i] = torch.tensor(self.point[i], dtype=torch.float32).to(device)
            if self.labels[i] is not None: self.labels[i] = torch.tensor(self.labels[i], dtype=torch.int64).to(device)
            if self.bboxes[i] is not None: self.bboxes[i] = torch.tensor(self.bboxes[i], dtype=torch.float32).to(device)


class ConcatBatcher(object):
    """
        ConcatBatcher selects batch generator according to selected model.
        Provides function collate_fn, which can be provided to torch.DataLoader
        to define how custom batching should be executed.
    """

    def __init__(self, device, model, box_params_num):
        """Initialize.
        Args:
            device: torch device 'gpu' or 'cpu'
        Returns:
            class: The corresponding class.
        """
        super(ConcatBatcher, self).__init__()
        self.device = device
        self.model = model
        self.box_params_num = box_params_num

    def collate_fn(self, batches):
        """Collate function called by original PyTorch dataloader.
        Args:
            batches: a batch of data
        Returns:
            class: the batched result
        """

        batching_result = ObjectDetectBatcher(batches, self.box_params_num)

        return batching_result


class TorchDataloader(Dataset):
    """
    This class allows you to load datasets for a PyTorch framework.

    Takes dataset object and performs operations defined with model (preprocessing, transformation, sampling) and
    it caches preprocess data if required.
    """

    def __init__(self,
                 dataset=None,
                 preprocess=None,
                 transform=None
                ):

        """Initialize.
        Args:
            dataset: The 3D ML dataset class. You can use the base dataset, sample datasets , or a custom dataset.
            preprocess: The model's pre-process method.
            transform: The model's transform method.
        Returns:
            class: The corresponding class.
        """
        self.dataset = dataset
        self.preprocess = preprocess
        self.transform = transform

    def __getitem__(self, index):

        """Returns the item at index position (idx)."""
        dataset = self.dataset
        index = index % len(dataset)
        
        attr = dataset.get_attr(index)

        # If datum just preprocessed
        if self.preprocess:
            data = self.preprocess(dataset.get_data(index), attr)
        # Nothnig happens
        else:
            data = dataset.get_data(index)

        # Transform datum if it's required
        if self.transform is not None:
            data = self.transform(data, attr)
        
        # Return data and attributes which are provided with Dataset3D class
        inputs = {'data': data, 'attr': attr}

        return inputs

    def __len__(self):
        """Returns the number of steps for an epoch."""

        steps_per_epoch = len(self.dataset)
            
        return steps_per_epoch