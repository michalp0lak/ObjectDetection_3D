import os.path
from addict import Dict
import yaml
import numpy as np
from utils import make_dir


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config(object):

    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict should be a dict, but'
                            f'got {type(cfg_dict)}')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))

        self.cfg_dict = cfg_dict

    def dump(self, *args, **kwargs):
        """Dump to a string."""

        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, ConfigDict):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self._cfg_dict, [])
        
        return self_as_dict

    @staticmethod
    def initialize_params(cfg):

        # merge device argument from global
        if cfg.global_args.device is not None:
            cfg.dataset.device = cfg.global_args.device
            cfg.pipeline.device = cfg.global_args.device
            cfg.model.device = cfg.global_args.device

        # Create output directory and merge this arg to pipeline
        if cfg.global_args.output_path is not None:

            make_dir(cfg.global_args.output_path)
            cfg.pipeline.log_dir = cfg.global_args.output_path

        # initialize cached dataset path
        if cfg.dataset.dataset_path is not None:
            cfg.dataset.cache_dir = os.path.join(cfg.dataset.dataset_path, 'cache/')

        # merge random seed argument from global and initialize random generator
        if cfg.global_args.seed is not None:

            rng = np.random.default_rng(cfg.global_args.seed)
            
            cfg.dataset.seed = cfg.global_args.seed
            cfg.pipeline.seed = cfg.global_args.seed
            cfg.model.seed = cfg.global_args.seed

            cfg.dataset.rng = rng
            cfg.pipeline.rng = rng
            cfg.model.rng = rng

        # Share model name, dataset name and box_parameters_number to pipeline cfg
        cfg.pipeline.model_name = cfg.model.name
        cfg.pipeline.dataset_name = cfg.dataset.name
        cfg.pipeline.box_params_num = cfg.global_args.box_params_num
        cfg.pipeline.eval_dim = cfg.global_args.model_dim

        # Initialize model parameters
        ## Initalize voxel grid dimensions
        cfg.model.voxel_grid_dims = ((np.array(cfg.model.point_cloud_range[3:]) - 
                                    np.array(cfg.model.point_cloud_range[:3])) / np.array(cfg.model.voxelize.voxel_size)).tolist()

        ## Initialize number of input features
        input_feat = cfg.model.get('input_features', None)

        if input_feat is not None:
            assert len(input_feat) >= 4, 'Not enough features, at least (x,y,z,reflectance) is required'
            assert sum([i in input_feat for i in range(4)]) == 4, 'Indexes (0,1,2,3) for (x,y,z, reflectance) has to be selected'
            #cfg.model.voxel_encoder['in_channels'] = len(input_feat)
        else:
            cfg.model.input_features = [0,1,2,3]

        ## Initialize vertical encoder params
        cfg.model.vertical_encoder.output_shape = [cfg.model.voxel_grid_dims [i] for i in [2,0,1]]

        ## Share params to model detection head
        cfg.model.head.box_params_num = cfg.global_args.box_params_num
        cfg.model.head.nms_dim = cfg.global_args.model_dim

        #cfg.model.head.ranges.append(cfg.model.point_cloud_range)

        return cfg.dataset, cfg.pipeline, cfg.model

    @staticmethod
    def merge_module_cfg_file(args):
        
        """
            Merge args and extra_dict from the input arguments.
            Merge the dict parsed by MultipleKVAction into this cfg.
        """
        # merge args to cfg
        cfg_dataset = Config.load_from_file(args.cfg_dataset)
        cfg_model = Config.load_from_file(args.cfg_model)
        cfg_pipeline = Config.load_from_file(args.cfg_pipeline)

        cfg_dict = {
            'dataset': cfg_dataset.cfg_dict,
            'model': cfg_model.cfg_dict,
            'pipeline': cfg_pipeline.cfg_dict
        }
        cfg = Config(cfg_dict)

        return Config.merge_cfg_file(cfg, args)

    @staticmethod
    def load_from_file(filename):
        
        if filename is None:
            raise FileExistsError("Config file is not defined")
        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} not found')

        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            raise ImportError('Config file has to yaml or yml file')

        else: 
            with open(filename) as f: cfg_dict = yaml.safe_load(f)

        return Config(cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)
