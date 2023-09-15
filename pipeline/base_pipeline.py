from abc import ABC, abstractmethod
import os
from datetime import datetime
import numpy as np
import torch
from config import Config
from utils import make_dir
from pipeline.utils import resume_training_process

class BasePipeline(ABC):
    """Base pipeline class."""

    def __init__(self, model, dataset, global_cfg, **kwargs):
        """Initialize.
        Args:
            model: A network model.
            dataset: A dataset, or None for inference model.
            device: 'gpu' or 'cpu'.
            kwargs:
        Returns:
            class: The corresponding class.
        """
        self.cfg = Config(kwargs)
        self.global_cfg = global_cfg

        if kwargs['name'] is None:
            raise KeyError("Please give a name to the pipeline")
        
        self.name = self.cfg.name
        self.model = model
        self.dataset = dataset
        self.rng = np.random.default_rng(kwargs.get('seed', None))


        if self.cfg.device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if len(self.cfg.device.split(':')) ==
                                       1 else 'cuda:' + self.cfg.device.split(':')[1])
        self.summary = {}
        self.cfg.setdefault('summary', {})

        if self.cfg.inference_mode:

            version = self.cfg.get('resume_from', None)

            if version is None:
                raise ValueError('There is no model defined in config file for inference (resume_from?)')
            else:

                assert type(version) == str, 'Invalid resume_from folder name format'
                assert bool(datetime.strptime(version, "%Y-%m-%d-%H-%M-%S")), 'Invalid ' 
                'resume_from folder name format'

                self.cfg.log_dir = os.path.join(self.cfg.log_dir, version + '/logs/')

        else:
        
            if self.cfg.is_resume:

                version = self.cfg.get('resume_from', None)

                if version: 

                    assert type(version) == str, 'Invalid resume_from folder name format'
                    assert bool(datetime.strptime(version, "%Y-%m-%d-%H-%M-%S")), 'Invalid ' 
                    'resume_from folder name format' 

                    self.cfg.log_dir = os.path.join(self.cfg.log_dir, version + '/logs/')
                
                else: 
                    version = resume_training_process(self.cfg.log_dir)
                    self.cfg.log_dir = os.path.join(self.cfg.log_dir, version + '/logs/')
                    
            else:
                self.cfg.log_dir = os.path.join(self.cfg.log_dir, 
                                                datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/logs/')

        make_dir(self.cfg.log_dir)

    @abstractmethod
    def run_inference(self, data):
        """Run inference on a given data.
        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        return

    @abstractmethod
    def run_testing(self):
        """Run testing on test sets."""
        return

    @abstractmethod
    def run_training(self):
        """Run training on train sets."""
        return

    @abstractmethod
    def show_inference(self):
        """Show inference on test sample."""
        return