import logging
from dataset.ForestDataset import Forest3D
from model.PointPillars import PointPillars
from pipeline.pipeline import ObjectDetection
from config import Config


def main():

    cfg = Config.load_from_file('./config.yaml')
    global_cfg = cfg.dump()

    if (cfg.global_args and cfg.pipeline and cfg.model and cfg.dataset) is None:
            raise ValueError("Please specify global arguments, pipeline, model, and dataset in config file")

    cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = Config.initialize_params(cfg)
    dataset = Forest3D(**cfg_dict_dataset)
    model = PointPillars(**cfg_dict_model)
    pipeline = ObjectDetection(model, dataset, global_cfg, **cfg_dict_pipeline)

    if not cfg_dict_pipeline.inference_mode:
        pipeline.run_training()
    else:
         raise ValueError("Can't run training session with configuration of inference_mode: True")

if __name__ == '__main__':

    logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',)

    main()