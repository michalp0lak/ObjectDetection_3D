from model.PointPillars import PointPillars
from dataset.ForestDataset import Forest3D
from pipeline.pipeline import ObjectDetection
from config import Config

def main():

    cfg = Config.load_from_file('./config.yaml')
    global_cfg = cfg.dump()
    
    if (cfg.global_args and cfg.pipeline and cfg.model and cfg.dataset) is None:    
        raise ValueError("Please specify global arguments, pipeline, model, and dataset in config file")
    
    cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = Config.initialize_params(cfg)

    model = PointPillars(**cfg_dict_model)
    dataset = Forest3D(cfg_dict_dataset.pop('dataset_path', None),**cfg_dict_dataset)
    pipeline = ObjectDetection(model, dataset, global_cfg,**cfg.pipeline)
   
    if cfg_dict_pipeline.get('inference_mode'):
        pipeline.show_inference()
    else:
        raise ValueError("Can't run show_inference session with configuration of inference_mode: False")

if __name__ == '__main__':

    main()