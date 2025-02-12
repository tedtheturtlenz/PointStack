from .. import heads, encoders
from .network_template import NetworkTemplate

import torch
import torch.nn.functional as F
import numpy as np


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y

class PointStack(NetworkTemplate):
    def __init__(self, cfg, topology=None):
        super().__init__(cfg, topology=topology)
        self.build_networks()
        self.cfg = cfg

    def get_loss(self, data_dic, smoothing=True, is_segmentation=False):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        if is_segmentation:
            pred_logits = data_dic['pred_score_logits'].contiguous().view(-1, self.cfg.DATASET.NUM_CLASS) # (Batch size * Num Points, Num Classes)
            gt_cls_id = data_dic['seg_id'].contiguous().view(-1, 1).long() # (Batch size * Num Points, 1)
        else:
            pred_logits = data_dic['pred_score_logits'] # (Batch size, Num Classes)
            gt_cls_id = data_dic['cls_id'] # (Batch size, 1)

        if smoothing:
            eps = 0.2
            n_class = pred_logits.size(1)
            one_hot = torch.zeros_like(pred_logits).scatter(1, gt_cls_id.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred_logits, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss[torch.isfinite(loss)].mean()
        else:
            loss = F.cross_entropy(pred_logits, gt_cls_id, reduction='mean')

        loss_dict = {
            'Cls': loss.item(),
        }

        return loss, loss_dict

    def compute_overall_iou(self, pred, target, num_classes):
        shape_ious = []
        pred = pred.max(dim=2)[1]    # (batch_size, num_points)
        pred_np = pred.cpu().data.numpy()
        target_np = target.cpu().data.numpy()

        for shape_idx in range(pred.size(0)):   # sample_idx
            part_ious = []
            for part in range(num_classes): 
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                F = np.sum(target_np[shape_idx] == part)

                if F != 0:
                    iou = I / float(U)    
                    part_ious.append(iou)   
            shape_ious.append(np.mean(part_ious))  
        return shape_ious   # [batch_size]


    def segment_predict(self, file_paths):
        '''
        Predicts the class for each point in multiple point clouds.
        
        Args:
            file_paths (list): A list of file paths to point cloud `.txt` files.
        
        Returns:
            torch.Tensor: The batched point clouds with the predicted class added as a new field.
        '''
        # Ensure the model is in evaluation mode
        self.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        batch_points = []
        batch_normals = []

        for file_path in file_paths:
            # Load point cloud (x, y, z, nx, ny, nz)
            data = np.loadtxt(file_path).astype(np.float32)
            points = data[:, :3]  # XYZ
            normals = data[:, 3:6]  # Normals NX, NY, NZ

            # Ensure we have exactly NUM_POINTS (resample if needed)
            num_points = 2048
            if points.shape[0] >= num_points:
                choice = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                choice = np.random.choice(points.shape[0], num_points, replace=True)

            batch_points.append(points[choice, :])
            batch_normals.append(normals[choice, :])

        # Convert to PyTorch tensor and move to device
        points_tensor = torch.from_numpy(np.stack(batch_points)).to(device)  # Shape: (batch_size, num_points, 3)
        normals_tensor = torch.from_numpy(np.stack(batch_normals)).to(device)  # Shape: (batch_size, num_points, 3)

        print(f"points_tensor shape: {points_tensor.shape}")  # Should print (batch_size, 2048, 3)

        # Construct data dictionary as expected by the network
        batch_size = points_tensor.shape[0]

        data_dic = {
            'points': points_tensor,  # Shape: (batch_size, num_points, 3)
            'norms': normals_tensor,  # Shape: (batch_size, num_points, 3)
        }

        # Create dummy class tokens for prediction
        cls_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=points_tensor.device)

        print(f"{cls_tokens.shape} is the shape of CLS")
        
        # Add the dummy cls_tokens to the data dictionary
        data_dic['cls_tokens'] = cls_tokens

        # Forward pass through the network
        with torch.no_grad():
            output = self.forward(data_dic)  # Pass the dictionary to the network
        
        # The output will be logits of shape (batch_size, num_points, num_classes)
        pred_logits = output['pred_score_logits']  # Shape: (batch_size, num_points, num_classes)

        # Convert logits to class predictions
        pred_classes = pred_logits.argmax(dim=-1)  # Shape: (batch_size, num_points)

        # Add predicted classes to the point cloud (append as new feature)
        tensor_point_cloud_with_predictions = torch.cat([points_tensor, pred_classes.unsqueeze(-1).float()], dim=-1)
        
         #Convert PyTorch tensor to NumPy
        point_cloud_np = tensor_point_cloud_with_predictions.cpu().numpy()  # Shape: (N, 4) -> XYZ + predicted label


        return point_cloud_np
