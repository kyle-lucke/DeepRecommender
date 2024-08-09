# Copyright (c) 2017 NVIDIA Corporation
import torch
import argparse
from reco_encoder.data import input_layer
from reco_encoder.model import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.autograd import Variable
import copy
import json
import time
import random
from pathlib import Path
from math import sqrt
import numpy as np
import os

def save_args(args_dict, save_fname):
    json.dump(args_dict, open(save_fname, 'w'))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='RecoEncoder')
parser.add_argument('--lr', type=float, default=0.00001, metavar='N',
                    help='learning rate')

parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N',
                    help='L2 weight decay')

parser.add_argument('--drop_prob', type=float, default=0.0, metavar='N',
                    help='dropout drop probability')

parser.add_argument('--noise_prob', type=float, default=0.0, metavar='N',
                    help='noise probability')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='global batch size')

parser.add_argument('--summary_frequency', type=int, default=100, metavar='N',
                    help='how often to save summaries')

parser.add_argument('--aug_step', type=int, default=-1, metavar='N',
                    help='do data augmentation every X step')

parser.add_argument('--constrained', action='store_true',
                    help='constrained autoencoder')

parser.add_argument('--skip_last_layer_nl', action='store_true',
                    help='if present, decoder\'s last layer will not apply non-linearity function')

parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                    help='maximum number of epochs')

parser.add_argument('--save_every', type=int, default=3, metavar='N',
                    help='save every N number of epochs')

parser.add_argument('--optimizer', type=str, default="momentum", metavar='N',
                    help='optimizer kind: adam, momentum, adagrad or rmsprop')

parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N',
                    help='hidden layer sizes, comma-separated')

parser.add_argument('--gpu_ids', type=str, default="0", metavar='N',
                    help='comma-separated gpu ids to use for data parallel training')

parser.add_argument('--path_to_train_data', type=str, default="", metavar='N',
                    help='Path to training data')

parser.add_argument('--path_to_eval_data', type=str, default="", metavar='N',
                    help='Path to evaluation data')

parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                    help='type of the non-linearity used in activations')

parser.add_argument('--logdir', type=str, default="logs", metavar='N',
                    help='where to save model and write logs')

parser.add_argument('--seed', type=int, default=0, help='Seed for RNGs')

args = parser.parse_args()
print(args)

set_seed(args.seed)

use_gpu = torch.cuda.is_available() # global flag
if use_gpu:
    print('GPU is available.') 
else: 
    print('GPU is not available.')

def do_eval(encoder, evaluation_data_layer):
  encoder.eval()
  denom = 0.0
  total_epoch_loss = 0.0
  for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
    inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
    targets = Variable(eval.cuda().to_dense() if use_gpu else eval.to_dense())
    outputs = encoder(inputs)
    loss, num_ratings = model.MSEloss(outputs, targets)
    total_epoch_loss += loss.item()
    denom += num_ratings.item()
  return sqrt(total_epoch_loss / denom)


def main():
  params = dict()
  params['batch_size'] = args.batch_size
  params['data_dir'] =  args.path_to_train_data
  params['major'] = 'users'
  params['itemIdInd'] = 1
  params['userIdInd'] = 0
  print("Loading training data")
  data_layer = input_layer.UserItemRecDataProvider(params=params)
  print("Data loaded")
  print("Total items found: {}".format(len(data_layer.data.keys())))
  print("Vector dim: {}".format(data_layer.vector_dim))

  print("Loading eval data")
  eval_params = copy.deepcopy(params)
  # must set eval batch size to 1 to make sure no examples are missed
  eval_params['data_dir'] = args.path_to_eval_data
  eval_data_layer = input_layer.UserItemRecDataProvider(params=eval_params,
                                                        user_id_map=data_layer.userIdMap, # the mappings are provided
                                                        item_id_map=data_layer.itemIdMap)
  eval_data_layer.src_data = data_layer.data
  rencoder = model.AutoEncoder(layer_sizes=[data_layer.vector_dim] + [int(l) for l in args.hidden_layers.split(',')],
                               nl_type=args.non_linearity_type,
                               is_constrained=args.constrained,
                               dp_drop_prob=args.drop_prob,
                               last_layer_activations=not args.skip_last_layer_nl)
  os.makedirs(args.logdir, exist_ok=True)
  model_checkpoint = args.logdir + "/model"
  path_to_model = Path(model_checkpoint)
  if path_to_model.is_file():
    print("Loading model from: {}".format(model_checkpoint))
    rencoder.load_state_dict(torch.load(model_checkpoint))

  save_args(vars(args), os.path.join(args.logdir, 'exp_args.json'))

  print('######################################################')
  print('######################################################')
  print('############# AutoEncoder Model: #####################')
  print(rencoder)
  print('######################################################')
  print('######################################################')

  best_val_rmse = float('inf')
  
  gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
  print('Using GPUs: {}'.format(gpu_ids))
  if len(gpu_ids)>1:
    rencoder = nn.DataParallel(rencoder,
                               device_ids=gpu_ids)
  
  if use_gpu: rencoder = rencoder.cuda()

  if args.optimizer == "adam":
    optimizer = optim.Adam(rencoder.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
  elif args.optimizer == "adagrad":
    optimizer = optim.Adagrad(rencoder.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
  elif args.optimizer == "momentum":
    optimizer = optim.SGD(rencoder.parameters(),
                          lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
  elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(rencoder.parameters(),
                              lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
  else:
    raise  ValueError('Unknown optimizer kind')

  t_loss = 0.0
  t_loss_denom = 0.0
  global_step = 0

  if args.noise_prob > 0.0:
    dp = nn.Dropout(p=args.noise_prob)

  for epoch in range(args.num_epochs):
    print('Doing epoch {} of {}'.format(epoch, args.num_epochs))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0
    for i, mb in enumerate(data_layer.iterate_one_epoch()):
      inputs = Variable(mb.cuda().to_dense() if use_gpu else mb.to_dense())
      optimizer.zero_grad()
      outputs = rencoder(inputs)
      loss, num_ratings = model.MSEloss(outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      global_step += 1
      t_loss += loss.item()
      t_loss_denom += 1

      if i % args.summary_frequency == 0:
        print('[%d, %5d] RMSE: %.7f' % (epoch, i, sqrt(t_loss / t_loss_denom)), flush=True)

        t_loss = 0
        t_loss_denom = 0.0

      total_epoch_loss += loss.item()
      denom += 1
      
      #if args.aug_step > 0 and i % args.aug_step == 0 and i > 0:
      if args.aug_step > 0:
        # Magic data augmentation trick happen here
        for t in range(args.aug_step):
          inputs = Variable(outputs.data)
          if args.noise_prob > 0.0:
            inputs = dp(inputs)
          optimizer.zero_grad()
          outputs = rencoder(inputs)
          loss, num_ratings = model.MSEloss(outputs, inputs)
          loss = loss / num_ratings
          loss.backward()
          optimizer.step()

    if args.optimizer == "momentum":
        scheduler.step()
  
    e_end_time = time.time()
    print('Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}'
          .format(epoch, e_end_time - e_start_time, sqrt(total_epoch_loss/denom)))

    if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
      eval_loss = do_eval(rencoder, eval_data_layer)

      print('Epoch {} EVALUATION LOSS: {}'.format(epoch, eval_loss))
      
      if eval_loss < best_val_rmse:

          out_fname = model_checkpoint + ".best"
          
          print(f'Validation loss improved from {best_val_rmse} to {eval_loss}, saving model. ')
          print(f"Saving model to {out_fname}")
          torch.save(rencoder.state_dict(), out_fname)

          best_val_rmse = eval_loss
          
  print("Saving model to {}".format(model_checkpoint + ".last"))
  torch.save(rencoder.state_dict(), model_checkpoint + ".last")

  # save to onnx
  dummy_input = Variable(torch.randn(params['batch_size'], data_layer.vector_dim).type(torch.float))
  torch.onnx.export(rencoder.float(), dummy_input.cuda() if use_gpu else dummy_input, 
                    model_checkpoint + ".onnx", verbose=True)
  print("ONNX model saved to {}!".format(model_checkpoint + ".onnx"))

if __name__ == '__main__':
  main()
