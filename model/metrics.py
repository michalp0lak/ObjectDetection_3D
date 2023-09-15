import numpy as np
import torch

class MetricEvaluator():

    def __init__(self, device, eval_dim):
        
        self.eval_dim = eval_dim
        self.device = device

    def filter_data(self, data, labels):
        """Filters the data to fit the given labels.
        Args:
            data (dict): Dictionary with the data (as numpy arrays).
                {
                    'label':      [...], # expected
                    ...
                }
            labels (number[]): List of labels which should be maintained.

        Returns:
            Tuple with dictionary with same as format as input, with only the given labels
            and difficulties and the indices.
        
        """
        # Mapping over whole data for each class and checking if data sample is in labels
        cond = torch.cat([data['label'].unsqueeze(-1) == label for label in labels if label is not None], 
                        axis = 1).reshape(-1,len(labels)).any(axis=1)

        idx = torch.where(cond)[0]
        result = {}

        # Filtering valid data samples
        for k in data:
            result[k] = data[k][idx]

        return result, idx

    def precision_recall_evaluation(self,
                                    pred,
                                    target,
                                    classes,
                                    min_overlap=[0.5]
                                ):

        """Computes precision quantities for each predicted box.
        Args:
            pred (dict): Dictionary with the prediction data (as numpy arrays).
                {
                    'bbox':       [...],
                    'label':      [...],
                    'score':      [...],
                    ...
                }
            target (dict): Dictionary with the target data (as numpy arrays).
                {
                    'bbox':       [...],
                    'label':      [...],
                    ...
                }
            classes (number[]): List of classes which should be evaluated.
                Default is [0].
            min_overlap (number[]): Minimal overlap required to match bboxes.
                One entry for each class expected. Default is [0.5].

        Returns:
            A tuple with a list of detection quantities
            (score, true pos., false. pos) for each box
            and a list of the false negatives.
        """

        if self.eval_dim == 3:
            from ops.ops_torch import bbox2corners3D as box2vertices
            from ops.ops_torch import box3d_overlap as box_overlap
        elif self.eval_dim == 2:
            from ops.ops_torch import bbox2rotated_corners2D as box2vertices
            from ops.ops_torch import bbox_iou2D as box_overlap

        # pre-filter data, remove unknown classes
        # data for unknown classes are removed
        pred, pred_idx = self.filter_data(pred, classes)
        target, target_idx = self.filter_data(target, classes)

        # For given batch IoU matrix between predictions and targets is evaluated
        detection = torch.zeros((len(classes), len(pred['bbox']), 3)).to(self.device)
        fns = torch.zeros((len(classes), 1), dtype=torch.int64).to(self.device)
        
        if pred['bbox'].shape[0] == 0:
            for i, label in enumerate(classes):
                fns[i] = len(self.filter_data(target, [label])[1])
            return detection, fns

        overlap = box_overlap(box2vertices(pred['bbox']), 
                              box2vertices(target['bbox']))

        # Output matrices:
            # detection stores for each class - all predicted bboxes with score, true positive indicator, false negative indicator
            # fns stores for each class count of false negatives based on min_overlap
        
        ## Evaluate for each class
        for i, label in enumerate(classes):

            # filter predicted data and indices for given class
            pred_label, pred_idx_l = self.filter_data(pred, [label])
            # filter ground truth data and indices for given class
            target_label, target_idx_l = self.filter_data(target, [label])

            # Get IoU matrix just for given class with prefiltered indices -> submatrix of overlap matrix
            overlap_label = overlap[pred_idx_l,:][:,target_idx_l]
            #overlap_label = overlap_label[:,target_idx_l]

            if len(overlap_label.shape) == 1: overlap_label = overlap_label.unsqueeze(-1)
            # if there are some predictions for given class
            if len(pred_idx_l) > 0:
                
                # no matching gt box (filtered preds vs all targets)
                false_positive = (overlap_label < min_overlap[i]).all(axis=1)
        
                # predicted bboxes which have IoU higher than min_overlap with target bboxes 
                # are considered as true positive
                match_cond = torch.any(overlap_label >= min_overlap[i],-1)
                # all matches first false positive
                false_positive[torch.where(match_cond)] = 1
            
                # Vector indicating if predicted box is true positive
                true_positive = torch.zeros((len(pred_idx_l),)).to(self.device)

                if len(target_idx_l) > 0:
                    # only best match can be true positive
                    # find for each target box index of predicted box having highest overlap with 
                    max_idx = torch.argmax(overlap_label, axis=0)

                    # Vector indicating predicted boxes which are best matches with target boxes
                    max_cond = torch.Tensor([True if idx in max_idx else False for idx in range(0,overlap_label.shape[0])]).to(self.device)
                else:
                    max_cond = torch.zeros(len(pred_idx_l)).to(self.device)

                # Potential true positives indexes && best matches indexes => true positives  
                global_cond = torch.logical_and(max_cond, match_cond)
                
                true_positive[global_cond] = 1
                false_positive[global_cond] = 0

                # False negatives - ground truth boxes which were not detected
                fns[i] = torch.sum(torch.all(overlap_label < min_overlap[i],axis=0))

                detection[i, pred_idx_l] = torch.stack([pred_label['score'], 
                                                        true_positive, false_positive], axis=-1)     
            else:

                fns[i] = len(target_idx_l)

        return detection, fns


    def evaluate(self,
                 pred,
                 target,
                 classes,
                 min_overlap
                ):

        """Computes precision and recall of the given prediction for batch.
        Args:
            pred (dict): List of dictionaries with the prediction data (as numpy arrays).
                {
                    'bbox':       [...],
                    'label':      [...],
                    'score':      [...],
                }[]
            target (dict): List of dictionaries with the target data (as numpy arrays).
                {
                    'bbox':       [...],
                    'label':      [...],

                }[]
            [torch.Tensor(data['label'] == label) for label in labels]
        """

        if len(min_overlap) != len(classes):
            assert len(min_overlap) == 1
            min_overlap = min_overlap * len(classes)
        assert len(min_overlap) == len(classes)

        cnt = 0
        box_cnts = [0]

        # Predicted counts per class stored in cumulative way
        for p in pred:
            cnt += len(self.filter_data(p, classes)[1])
            box_cnts.append(cnt) 

        # Ground truth counts per class
        gt_cnt = torch.zeros((len(classes))).to(self.device)
        for i, c in enumerate(classes):
                for t in target:
                    gt_cnt[i] += len(self.filter_data(t, [c])[1])

        detection = torch.zeros((len(classes),  box_cnts[-1], 3)).to(self.device)
        fns = torch.zeros((len(classes), 1), dtype=torch.int64).to(self.device)

        # For each item in batch
        for i in range(len(pred)):

            # Compute true/false positives
            d, f = self.precision_recall_evaluation(pred=pred[i],
                                target=target[i],
                                classes=classes,
                                min_overlap=min_overlap)
            
            detection[:, box_cnts[i]:box_cnts[i + 1]] = d
            fns += f

        # Matrix to store precision and recall for each class
        recall = torch.zeros((len(classes), 1)).to(self.device)
        precision = torch.zeros((len(classes), 1)).to(self.device)

        # for every class
        for i in range(len(classes)):

            # Recall
            recall[i] = 100 * (detection[i,:,1].sum()/(detection[i,:,1].sum()+fns[i]))

            # Precision
            precision[i] = 100 * (detection[i,:,1].sum()/(detection[i,:,1].sum()+detection[i,:,2].sum()))

        return precision.cpu().detach().numpy(), recall.cpu().detach().numpy()