import torch
import torch.nn as nn

class BCECustomLoss(nn.Module):
    def __init__(self, weight=None):
        super(BCECustomLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(weight=self.weight)
        return bce_loss(inputs, targets)

class DFLLoss(nn.Module):
    def forward(self, S_i, S_i1, y):
        loss = -((y - S_i) * torch.log(S_i) + (S_i1 - y) * torch.log(S_i1))
        return loss

class WiseIoULoss(nn.Module):
    def __init__(self, alpha, beta, delta, gamma):
        super(WiseIoULoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

    def forward(self, x_p, y_p, x_gt, y_gt, W_p, H_p, W_gt, H_gt):
        # Calculate Wise-IoU loss
        S_u = W_p * H_p + W_gt * H_gt - self.beta * self.gamma
        loss = (1 - self.beta * self.alpha * self.beta * self.alpha * self.gamma) * torch.exp(
            ((x_p - x_gt) ** 2 + (y_p - y_gt) ** 2) / (self.beta ** 2 + self.gamma ** 2)) ** self.delta
        return loss

class CustomLoss(nn.Module):
    def __init__(self, lambda1, lambda2, lambda3, alpha, beta, delta, gamma):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.bce_loss = BCECustomLoss()
        self.dfl_loss = DFLLoss()
        self.wise_iou_loss = WiseIoULoss(alpha, beta, delta, gamma)

    def forward(self, preds, targets):
        # Split predictions into classification and regression outputs
        classification_preds, regression_preds = preds

        # Extract targets for classification and regression
        classification_targets, regression_targets = targets

        # Calculate individual losses
        bce_loss = self.bce_loss(classification_preds, classification_targets)
        dfl_loss = self.dfl_loss(S_i, S_i1, classification_targets)
        wise_iou_loss = self.wise_iou_loss(x_p, y_p, x_gt, y_gt, W_p, H_p, W_gt, H_gt)

        # Combine losses with weights
        total_loss = self.lambda1 * bce_loss + self.lambda2 * dfl_loss + self.lambda3 * wise_iou_loss
        return total_loss
