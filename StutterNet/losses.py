import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss

class CCCLoss(nn.Module):
  '''concordance correlation coefficient loss'''
  def __init__(self, eps=1e-7):
      '''
      Args:
        eps (float, optional): stabilizing term
      '''
      super(CCCLoss, self).__init__()
      self.eps = eps
  def forward(self, y_hat, y):
      gold_mean = torch.mean(y.T, dim=-1, keepdim=True)
      pred_mean = torch.mean(y_hat.T, dim=-1, keepdim=True)
      covariance = (y.T-gold_mean)*(y_hat.T-pred_mean)
      gold_var = torch.mean(torch.square(y.T-gold_mean), dim=-1,  keepdim=True)
      pred_var = torch.mean(torch.square(y_hat.T-pred_mean), dim=-1, keepdim=True)
      ccc = 2 * covariance / (gold_var + pred_var + torch.square(gold_mean - pred_mean) + self.eps)
      return torch.mean(1-ccc, dim=-1)
      # return torch.mean(torch.mean(1-ccc, dim=-1))

class SigmoidFocalLoss(nn.Module):
  def __init__(self, reduction=None):
    super(SigmoidFocalLoss, self).__init__()
    self.reduction = reduction

  def forward(self, y_hat , y):
    loss = sigmoid_focal_loss(y_hat, y, reduction=self.reduction)
    return loss

class StutterLoss(nn.Module):
  '''SEP-28k Loss '''
  def __init__(self, alpha=1, beta=1, stutter_weights=None, reduction='mean'):
    super(StutterLoss, self).__init__()
    self.stutter_loss = CCCLoss()
    self.disfluency_loss = SigmoidFocalLoss(reduction=reduction)
    self.alpha = alpha
    self.beta = beta
    self.stutter_weights = stutter_weights
    if (isinstance(self.stutter_weights, torch.Tensor)):
      self.stutter_weights = self.stutter_weights.reshape((1,-1))
  
  def forward(self, y_hat , y):
    '''expects list of inputs and outputs'''
    y_class, y_bin = torch.split(y, [6,6], dim=-1)
    y_hat_class, y_hat_bin = torch.split(y_hat, [6,6], dim=-1)
    disfluency_loss = self.disfluency_loss(y_hat_class, y_class)
    stutter_loss = torch.mean(self.stutter_loss(y_hat_bin, y_bin))
    if (not isinstance(self.stutter_weights, torch.Tensor)):
      return self.alpha * stutter_loss + self.beta * torch.mean(disfluency_loss, dim=0)
    return self.alpha * stutter_loss + self.beta * self.stutter_weights@disfluency_loss(y_hat_class, y_class)