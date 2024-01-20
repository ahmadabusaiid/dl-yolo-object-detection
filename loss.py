import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted

        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[...,-9:-5], predictions[...,-9:-5])
        iou_b2 = intersection_over_union(predictions[...,-4:], predictions[...,-4:])
        
        # torch.max returns the max value of the two tensors
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox is 0 or 1

        exists = target[..., self.C].unsqueeze(3) # Iobj_i
        

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        
        box_predictions = exists * (
            (
                bestbox * predictions[..., -4:]
                + (1 - bestbox) * predictions[..., -9:-5]
            )
        )

        box_targets = exists * target[..., -9:-5]
        
        # for slide: add 1e-6 for stability if sqrt is 0 the derivative is undefined
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) --> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        
        # ======================== #
        #   FOR OBJECT LOSS        #
        # ======================== #

        pred_box = (
            bestbox * predictions[..., -5:-4] + (1 - bestbox) * predictions[..., -10:-9]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists * pred_box),
            torch.flatten(exists * target[..., -10:-9]),
        )
        
        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #

        # (N, S, S, 1) --> (N*S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists) * predictions[..., -10:-9], start_dim=1),
            torch.flatten((1 - exists) * target[..., -10:-9], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists) * predictions[..., -5:-4], start_dim=1),
            torch.flatten((1 - exists) * target[..., -10:-9], start_dim=1),
        )
        
        # ======================== #
        #   FOR CLASS LOSS         #
        # ======================== #

        # (N, S, S, 20) --> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists * target[..., :self.C], end_dim=-2),
        )
        
        # ======================== #
        #   FINAL LOSS             #
        # ======================== #

        loss = (
            self.lambda_coord * box_loss  # first two rows of loss in paper
            + object_loss  # third row of loss in paper
            + self.lambda_noobj * no_object_loss  # forth row of loss in paper
            + class_loss  # fifth row of loss in paper
        )
        
        return loss