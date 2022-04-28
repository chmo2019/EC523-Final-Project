from sklearn.metrics import f1_score
import numpy as np

#TODO: implement as nn.Module subclass
  
def f1(y_hat, y):
  per_class_score = f1_score(y.cpu().detach().numpy().astype('int'), 
                  (sigmoid(y_hat.cpu().detach().numpy()) > 0.5).astype('int'),
                  average='samples', zero_division=1)
  return np.mean(per_class_score)

def accuracy(outputs, labels):
  # y_hat = (sigmoid(outputs.cpu().detach().numpy()).flatten() > 0.5).astype('int')
  # y = labels.cpu().detach().numpy().flatten().astype('int')
  y_hat = (sigmoid(outputs.cpu().detach().numpy()) > 0.5).astype('int')
  y = labels.cpu().detach().numpy().astype('int')
  batch_size = y.shape[0]
  per_class_acc = np.sum(y == y_hat, axis=0) / batch_size
  # total = float(len(y))
  # correct = float(np.sum(y == y_hat))
  # return correct / total
  return np.mean(per_class_acc)
