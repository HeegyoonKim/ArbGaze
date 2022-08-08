import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-8)
    sim = F.hardtanh(sim, -1.0 + 1e-8, 1.0 - 1e-8)
    return torch.acos(sim) * (180 / np.pi)


class GazeAngularLoss(object):
    def __call__(self, gaze, gaze_hat):
        y = gaze.detach()
        y_hat = gaze_hat
        return torch.mean(nn_angular_distance(y, y_hat))


class CosineSimilarityLoss(object):
    def __call__(self, a, b):
        return 1 - torch.sum(torch.mul(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2)))


class JS_DivergenseLoss(object):
    def __call__(self, a, b):
        loss = F.kl_div(F.log_softmax(a, dim=1), F.softmax(b, dim=1), reduction='batchmean') + \
               F.kl_div(F.log_softmax(b, dim=1), F.softmax(a, dim=1), reduction='batchmean')
        return loss


class KDLoss(object):
    def __init__(self, loss_type='L1'):
        if loss_type == 'JS':
            self.loss_fn = JS_DivergenseLoss()
        elif loss_type == 'COS':
            self.loss_fn = CosineSimilarityLoss()
        elif loss_type == 'L1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'MSE':
            self.loss_fn = nn.MSELoss(reduction='mean')
    
    def __call__(self, teacher_feats, student_feats):
        loss = 0.0
        for i in range(len(teacher_feats)):
            loss += self.loss_fn(teacher_feats[i].detach(), student_feats[i])
        return loss