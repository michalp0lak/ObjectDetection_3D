import logging
import re
from datetime import datetime
import os 
from os.path import join
import yaml
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import open3d as o3d

from torch.utils.data import DataLoader
from dataset.dataloaders import TorchDataloader, ConcatBatcher
from pipeline.utils import latest_ckpt
from utils import make_dir
from model.metrics import MetricEvaluator
from pipeline.base_pipeline import BasePipeline

log = logging.getLogger(__name__)

class ObjectDetection(BasePipeline):
    """Pipeline for object detection."""

    def __init__(self, model, dataset, global_cfg, **kwargs):

        super().__init__(model=model,
                         dataset=dataset,
                         global_cfg=global_cfg,
                         **kwargs)
                         
        self.ME = MetricEvaluator(self.device, self.cfg.eval_dim)

    def save_ckpt(self, epoch, save_best = False):

        ckpt_dir = join(self.cfg.log_dir, 'checkpoint/')
        make_dir(ckpt_dir)

        if save_best: path = join(ckpt_dir,'ckpt_best.pth')
        else: path = join(ckpt_dir, f'ckpt_{epoch:05d}.pth')

        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
                 path)

        log.info(f'Epoch {epoch:3d}: save ckpt to {path:s}')

    def load_ckpt(self):
        
        ckpt_dir = join(self.cfg.log_dir, 'checkpoint/')
        epoch = 0

        if not self.cfg.inference_mode:
            
            if self.cfg.is_resume:

                last_ckpt_path = latest_ckpt(ckpt_dir)
            
                if last_ckpt_path:

                    epoch = int(re.findall(r'\d+', last_ckpt_path)[-1]) + 1
                    ckpt_path = last_ckpt_path
                    log.info('Model restored from the latest checkpoint: {}'.format(epoch))

                else:

                    log.info('Latest checkpoint was not found')
                    log.info('Initializing from scratch.')
                    return epoch, None

            else:
                log.info('Initializing from scratch.')
                return epoch, None    

        else:

            ckpt_path = self.cfg.log_dir + 'checkpoint/ckpt_best.pth'

            if not os.path.exists(ckpt_path):
                raise ValueError('There is not pretrained model for inference. Best output of training should be found as {}'.format(ckpt_path))


        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device) 

        self.model.load_state_dict(ckpt['model_state_dict'])

        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info('Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        return epoch, ckpt_path

    def transform_box_batch(self, boxes, labels=None):

        dicts = []
       
        if labels is None:
            for box in boxes:dicts.append(box)
        else:
            for bbox, label in zip(boxes, labels):
                dicts.append({'bbox': bbox, 'label': label})

        return dicts

    def transform_for_metric(self, bboxes):

        """Convert data for evaluation:
        Args:
            bboxes: List of 3D bboxes.
        """

        box_dicts = {
            'bbox': torch.empty((len(bboxes), self.cfg.box_params_num)).to(self.device),
            'label': torch.empty((len(bboxes),)).to(self.device),
            'score': torch.empty((len(bboxes),)).to(self.device)
            }

        for i in range(len(bboxes)):
            box_dict = bboxes[i]

            for k in box_dict:
                box_dicts[k][i] = box_dict[k]

        return box_dicts


    def run_inference(self, data):
        """Run inference on given data.
        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """

        self.load_ckpt()
        self.model.eval()

        # If run_inference is called on raw data.
        if isinstance(data, dict):
            batcher = ConcatBatcher(self.device, self.cfg.model_name, self.cfg.box_params_num)
            data = batcher.collate_fn([{
                'data': data['data'],
                'attr': data['attr']
            }])

        data.to(self.device)

        with torch.no_grad():

            results = self.model(data)
            boxes = self.model.inference_end(results)

        return boxes

    def show_inference(self):

        test_dataset = self.dataset.get_split('test')
        test_split = TorchDataloader(dataset=test_dataset,
                                preprocess=self.model.preprocess,
                                transform=self.model.transform
                                )

        idx = random.sample(range(0, len(test_dataset)), 1)
        print(idx)
   
        #data = test_dataset.get_data(idx[0])
        data_item = test_split.__getitem__(idx[0])
        print(test_dataset.get_attr(idx[0]))
        results = self.run_inference(data_item)

        data = data_item['data']          
        target = [self.transform_for_metric(self.transform_box_batch(boxes = torch.Tensor(data['bboxes'].tolist()), 
                                        labels=torch.Tensor(data['labels'].tolist())))]
            
        prediction = [self.transform_for_metric(self.transform_box_batch(boxes)) for boxes in results]

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                 target,
                 self.model.classes_ids,
                 self.cfg.get("overlaps", [0.5]))

        print("")
        print(f' {" ": <9} "==== Precision ==== Recall ==== F1 ====" ')
       
        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        print("Overall_precision: {:.2f}".format(precision))
        print("Overall_recall: {:.2f}".format(recall ))
        print("F1: {:.2f}".format(f1))

        geometries = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data['point'][:,:3])
        geometries.append(pcd)

        for bbx in results[0]:

            box = bbx['bbox'].detach().cpu().numpy()
            rot_matx = np.array([[1.0, 0, 0],[0, np.cos(box[6]), -np.sin(box[6])],[0,np.sin(box[6]),np.cos(box[6])]])
            rot_maty = np.array([[np.cos(box[7]),0,np.sin(box[7])], [0, 1.0, 0],[-np.sin(box[7]),0,np.cos(box[7])]])
            rot_matz = np.array([[np.cos(box[8]), -np.sin(box[8]),0],[np.sin(box[8]), np.cos(box[8]), 0],[0, 0, 1.0]])
            rot_mat = np.dot(np.dot(rot_matz,rot_maty),rot_matx)

            o3box = o3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
            o3box.color = (1, 0, 0)
            geometries.append(o3box)

        for box in data['bboxes']:

            box[2] = box[2] + box[5]/2

            rot_matx = np.array([[1.0, 0, 0],[0, np.cos(box[6]), -np.sin(box[6])],[0,np.sin(box[6]),np.cos(box[6])]])
            rot_maty = np.array([[np.cos(box[7]),0,np.sin(box[7])], [0, 1.0, 0],[-np.sin(box[7]),0,np.cos(box[7])]])
            rot_matz = np.array([[np.cos(box[8]), -np.sin(box[8]),0],[np.sin(box[8]), np.cos(box[8]), 0],[0, 0, 1.0]])
            rot_mat = np.dot(np.dot(rot_matz,rot_maty),rot_matx)

            o3box = o3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
            o3box.color = (0, 1, 0)
            geometries.append(o3box)

        o3d.visualization.draw_geometries(geometries)


    def run_testing(self):
        """Run test with test data split, computes mean average precision of the
        prediction results.
        """

        test_folder = self.cfg.log_dir + "test/"
        make_dir(test_folder)

        self.model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(self.device))
        log_file_path = join(test_folder, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(self.device, self.cfg.model_name, self.cfg.box_params_num)

        test_split = TorchDataloader(dataset=self.dataset.get_split('testing'),
                                     preprocess=self.model.preprocess,
                                     transform=self.model.transform,
                                    )

        testing_loader = DataLoader(
            test_split,
            batch_size=self.cfg.testing_batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', True),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        self.load_ckpt()

        log.info("Started testing")
        prediction = []
        target = []

        with torch.no_grad():
            for data in tqdm(testing_loader, desc='testing'):

                data.to(self.device)

                if data.point[0].shape[0] > 0:

                    results = self.model(data)
                    boxes_batch = self.model.inference_end(results)
                              
                    target.extend([self.transform_for_metric(self.transform_box_batch(boxes, labels=labels)) 
                               for boxes, labels in zip(data.bboxes, data.labels)])
    
                    prediction.extend([self.transform_for_metric(self.transform_box_batch(boxes)) 
                                 for boxes in boxes_batch])
                else:
                    log.info("Invalid point cloud load: {}".format(data.attr[0]['path']))

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))

        log.info("")
        log.info(f' {" ": <9} "==== Precision ==== Recall ==== F1 ====" ')
        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]
            f1 = 2*p*rec/(p+rec)
            log.info(f' {"{}".format(c): <15} {"{:.2f}".format(p): <15.5} {"{:.2f}".format(rec): <10} {"{:.2f}".format(f1)}')

        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        log.info("")
        log.info("Overall_precision: {:.2f}".format(precision))
        log.info("Overall_recall: {:.2f}".format(recall ))
        log.info("F1: {:.2f}".format(f1))

        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)
        
        test_protocol = {
            '0_model': self.cfg.get('model_name', None),
            '1_model_version':self.cfg.get('resume_from', None), 
            '2_dataset': self.cfg.get('dataset_name', None), 
            '3_date': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), 
            '4_precision': precision, 
            '5_recall': recall, 
            '6_f1': f1
                    }

        with open(test_folder + 'test_protocol.yaml', 'w') as outfile:
            yaml.dump(test_protocol, outfile)

    def run_valid(self):
        """Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.
        Args:
            epoch (int): step for TensorBoard summary. Defaults to 0 if
                unspecified.
        """

        # Model in evaluation mode -> no gradient = parameters are not optimized
        self.model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        batcher = ConcatBatcher(self.device, self.cfg.model_name, self.cfg.box_params_num)

        valid_dataset = self.dataset.get_split('validation')

        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform,
                                     )

        validation_loader = DataLoader(
            valid_split,
            batch_size=self.cfg.validation_batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        log.info("Started validation")

        self.valid_losses = {}

        prediction = []
        target = []
        
        with torch.no_grad():
            for data in tqdm(validation_loader, desc='validation'):
                data.to(self.device)

                if data.point[0].shape[0] > 0:
                    results = self.model(data)
                    loss = self.model.loss(results, data)

                    for l, v in loss.items():
                        if l not in self.valid_losses:
                            self.valid_losses[l] = []
                        self.valid_losses[l].append(v.cpu().numpy())

                    # convert to bboxes for mAP evaluation
                    boxes_batch = self.model.inference_end(results)
                    
                    target.extend([self.transform_for_metric(self.transform_box_batch(boxes, labels=labels)) 
                               for boxes, labels in zip(data.bboxes, data.labels)])
    
                    prediction.extend([self.transform_for_metric(self.transform_box_batch(boxes)) 
                                 for boxes in boxes_batch])

                else:
                    log.info("Invalid point cloud load: {}".format(data.attr[0]['path']))

        # Process bar data feed
        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)

        desc += " > loss: %.03f" % sum_loss
        log.info(desc)
        
        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))

        log.info("")
        log.info(f' {" ": <9} "==== Precision ==== Recall ==== F1 ====" ')
        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]
            f1 = 2*p*rec/(p+rec)
            log.info(f' {"{}".format(c): <15} {"{:.2f}".format(p): <15.5} {"{:.2f}".format(rec): <10} {"{:.2f}".format(f1)}')

        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        log.info("")
        log.info("Overall_precision: {:.2f}".format(precision))
        log.info("Overall_recall: {:.2f}".format(recall ))
        log.info("F1: {:.2f}".format(f1))
        
        self.valid_losses["precision"] = precision
        self.valid_losses["recall"] = recall
        self.valid_losses["f1"] = f1

        return self.valid_losses


    def run_training(self):

        with open(self.cfg.log_dir + 'process_config.json', "w") as outfile:
            json.dump(dict(self.global_cfg), outfile)

        """Run training with train data split."""
        torch.manual_seed(self.rng.integers(np.iinfo(np.int32).max))  # Random reproducible seed for torch

        log.info("DEVICE : {}".format(self.device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(self.cfg.log_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(self.device, self.cfg.model_name, self.cfg.box_params_num)

        train_dataset = self.dataset.get_split('training')
        
        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform
                                     )
        
        train_loader = DataLoader(
            train_split,
            batch_size=self.cfg.training_batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        self.optimizer = self.model.get_optimizer(self.cfg.optimizer)

        start_ep, _ = self.load_ckpt()

        if os.path.exists(self.cfg.log_dir + '/training_record.csv'):
            training_record = pd.read_csv(self.cfg.log_dir + '/training_record.csv', index_col=False)
        else:
            training_record = pd.DataFrame([],columns=['epoch', 'precision', 'recall', 'f1'])

        log.info("Started training")
        for epoch in range(start_ep, self.cfg.max_epoch + 1):
            
            log.info(f'================================ EPOCH {epoch:d}/{self.cfg.max_epoch:d} ================================')
            self.model.train()
            self.losses = {}

            process_bar = tqdm(train_loader, desc='training')
            for data in process_bar:

                data.to(self.device)
                results = self.model(data)

                loss = self.model.loss(results, data)
                loss_sum = sum(loss.values())
                
                self.optimizer.zero_grad()
                loss_sum.backward()

                if self.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    self.cfg.grad_clip_norm)
                self.optimizer.step()

                desc = "training - "
                for l, v in loss.items():
                    if l not in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().detach().numpy())
                    desc += " %s: %.03f" % (l, v.cpu().detach().numpy())

                desc += " > loss: %.03f" % loss_sum.cpu().detach().numpy()
                process_bar.set_description(desc)
                process_bar.refresh()

            if os.path.exists(self.cfg.log_dir + '/metrics.npy'):
                metrics = np.load(self.cfg.log_dir + '/metrics.npy')
                best_f1 = metrics[2]
            else:
                best_f1 = 0
            # --------------------- validation of epoch -> given by self.run_valid()
            if (epoch % self.cfg.get("validation_freq", 1)) == 0:

                metrics = self.run_valid()

                training_record.loc[epoch] = [epoch,metrics['precision'], metrics['recall'], metrics['f1']]
                actual_f1 = metrics['f1']
                
                if actual_f1 > best_f1:

                    best_f1 = actual_f1
                    self.save_ckpt(epoch, save_best=True)
                    np.save(self.cfg.log_dir + '/metrics.npy', 
                            np.array([metrics['precision'], metrics['recall'], metrics['f1']]))

            if epoch % self.cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch,save_best=False)

            training_record.to_csv(self.cfg.log_dir + '/training_record.csv', index=False)