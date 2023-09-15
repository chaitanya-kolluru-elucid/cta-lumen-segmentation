import torch.nn as nn
from monai.losses import DiceCELoss
import torch

class deep_supervision(nn.Module):

    def __init__(self, include_background, to_onehot_y, softmax, batch, ce_weight):
        
        super(deep_supervision, self).__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.batch = batch
        self.ce_weight = ce_weight

        self.loss_function = DiceCELoss(include_background=self.include_background, to_onehot_y=self.to_onehot_y, softmax=self.softmax,
                                        batch=self.batch, ce_weight = self.ce_weight)

    def forward(self, y_pred, y_true):

        batch_loss = 0

        for batch_index in range(y_pred.shape[0]):
            preds = y_pred[batch_index,...]
            target = y_true[batch_index,...]

            sample_loss = 0

            for scale_index in range(preds.shape[0]):

                pred_at_current_scale = torch.unsqueeze(preds[scale_index,...], axis=0)
                target_image = torch.unsqueeze(target, axis=0)
                
                sample_loss += (0.5 ** scale_index) * self.loss_function(pred_at_current_scale, target_image)

            batch_loss += sample_loss

        return batch_loss