import torch
from torch import nn
import numpy as np
import argparse

def parser():
	#TODO: create parser
	ap = argparse.ArgumentParser()
	return ap.parse_args()

def train(net, trainloader, criterion, batch_size,
          validationloader=None, optimizer=None,
          scheduler=None, epochs=50, logdir=None, metrics=None,
          verbose=True, tuner=False, checkpoint_dir=None):
  ''' training loop function for simple
  supervised learning task.

  Args:
    net (torch.nn.Module): network to train
    trainloader (torch.utils.data.DataLoader): 
      train data loader
    criterion (torch.nn.object): criterion with which
      to optimize the provided network
    batch_size (int): batch of trainloader and validationloader
    validationloader (torch.utils.data.DataLoader, optional): 
      validation data loader
    optimizer (torch.optim.Optimizer, optional): 
      optimizer function, defaults to torch.nn.optim.Adam w/ amsgrad
    scheduler (torch.optim.lr_scheduler, optional):
      learning rate scheduler object
    epochs (int, optional): number of epochs to train network,
      defaults to 50
    logdir (string, optional): path to tensorboard log directory,
      if None logging default to ./runs/ directory
    metrics (list of tuples, optional): metrics to be logged with
      name and metric being the first and second element of the
      each tuple respectively
    verbose (bool, optional): whether or not to print information
      to console
    tuner (bool, optional): whether to employ ray tune
  '''
  from torch.utils.tensorboard import SummaryWriter
  from sklearn.metrics import classification_report
  writer = SummaryWriter(log_dir=logdir)

  if (verbose):
    from tensorflow.keras.utils import Progbar

  if (optimizer is None):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, amsgrad=True)
    
  if (checkpoint_dir is not None):
    state, optim_state = torch.load(os.path.join(
        checkpoint_dir, "checkpoint"))
    net.load_state_dict(state)
    optimizer.load_state_dict(optim_state)

  assert epochs > 0, "Assertion failed. epochs must be greater than 0!"

  steps = 0

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device

  net.train(True)
    
#   net.to(device)
    
  if (tuner):
    from ray import tune
    import os

  for i in range(epochs):
    num_batches = len(trainloader)
    num_samples = num_batches * batch_size

    if (verbose):
      print("\nepoch {}/{}".format(i+1,epochs))
      pbar = Progbar(target=num_batches)

    # if (metrics is not None):
    #   train_metrics = [0 for metric in metrics]

    y_true = np.zeros((num_samples, 12))
    y_pred = np.zeros((num_samples, 12))
    idx = 0

    for j, data in enumerate(iter(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data[0].to(device), [data[1][0].to(device), data[1][1].to(device)]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

        y_true[idx:idx+outputs.shape[0], :] = labels.detach().cpu().numpy()
        y_pred[idx:idx+outputs.shape[0], :] = outputs.detach().cpu().numpy()

        idx += outputs.shape[0]

        if (scheduler is not None):
            scheduler.step()

        if (verbose):
          pbar.update(j, values=[("loss", 
            train_loss.detach().cpu().numpy().item())])

        steps += 1 

        writer.add_scalar('Loss/train', 
          train_loss.detach().cpu().numpy().item(), steps)

        # if (metrics is not None):
          # for (j, metric) in enumerate(metrics):
          #   # train_metrics[j] += metric[1](outputs, labels).detach().cpu().numpy()
          #   train_metrics[j] += metric[1](outputs, labels)

    rep = classification_report(y_true.astype('int'), 
      (sigmoid(y_pred) > 0.5).astype('int'), target_names=target_names,
      output_dict=True)
    
    for k in rep.keys():
      for j in rep[k].keys():
        writer.add_scalar(j + '/' + k + '/train',
          rep[k][j], steps)

    # if (metrics is not None):
    #   for (j, metric) in enumerate(metrics):
    #     # writer.add_scalar(metric[0] + '/train', 
    #     #   train_metrics[j] / num_samples, steps)
    #     writer.add_scalar(metric[0] + '/train', 
    #       train_metrics[j] / num_batches, steps)

    if (validationloader is not None):
      net.train(False)
      val_loss = 0
      # if (metrics is not None):
      #   val_metrics = [0 for metric in metrics]
      num_val_batches = len(validationloader)
      num_val_samples = num_val_batches * batch_size

      y_val_true = np.zeros((num_val_samples, 12))
      y_val_pred = np.zeros((num_val_samples, 12))

      idx = 0

      for data in iter(validationloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # inputs, labels = data[0].to(device), [data[1][0].to(device), data[1][1].to(device)]
            
            outputs = net(inputs)
            val_loss += criterion(outputs, labels).detach().cpu().numpy()

            y_val_true[idx:idx+outputs.shape[0], :] = labels.detach().cpu().numpy()
            y_val_pred[idx:idx+outputs.shape[0], :] = outputs.detach().cpu().numpy()

            idx += outputs.shape[0]

            # if (metrics is not None):
            #     for (j, metric) in enumerate(metrics):
            #         # val_metrics[j] += metric[1](outputs, labels).detach().cpu().numpy()
            #         val_metrics[j] += metric[1](outputs, labels)
                    
      val_loss /= (num_val_batches) # assume all validation set used
      # scheduler.step(val_loss)

      rep = classification_report(y_val_true.astype('int'), 
        (sigmoid(y_val_pred) > 0.5).astype('int'), target_names=target_names,
        output_dict=True)
      print(classification_report(y_val_true.astype('int'), 
                                  (sigmoid(y_val_pred) > 0.5).astype('int'), target_names=target_names))
      #  output_dict=False)
      #print(rep2)
      
      for k in rep.keys():
        for j in rep[k].keys():
          writer.add_scalar(j + '/' + k + '/valid',
            rep[k][j], steps)

      writer.add_scalar('Loss/valid', val_loss, steps)

      # if (metrics is not None):
      #   for (j, metric) in enumerate(metrics):
      #     # writer.add_scalar(metric[0] + '/valid', 
      #     #   val_metrics[j] / num_val_samples, steps)
      #      writer.add_scalar(metric[0] + '/valid', 
      #       val_metrics[j] / num_val_batches, steps)
        
      # if (tuner):
      #   with tune.checkpoint_dir(i+1) as checkpoint_dir:
      #       path = os.path.join(checkpoint_dir, "checkpoint")
      #       torch.save((net.state_dict(), optimizer.state_dict()), path)
            
      #   tune.report(loss=val_loss, accuracy=val_metrics[0] / num_val_samples, iters=i+1)
          
      if (verbose):
        pbar.update(num_batches, values=[("val_loss",val_loss.item())])
      net.train(True)
    else:
      if (verbose):
        pbar.update(num_batches, values=None)

if __name__ == "__main__":
	args = parser() # get arguments

	# TODO: implement args such that we can train from the command line
	#train(args.net, args.trainloader, args.criterion, args.batch_size,
        #  args.validationloader, args.optimizer,
        #  args.scheduler, args.epochs, args.logdir, args.metrics,
        #  args.verbose, args.tuner, args.checkpoint_dir):
