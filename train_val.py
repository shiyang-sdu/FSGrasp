from models.pointnet import PointNetPP
from models.mlp import MLPModel
from models.pointnet_attention import PointNetPP_A
from models.pointkan import PointKAN
from models.pointkan_attention import PointKAN_A
# from models.pointweb_module import PointWebSAModule
from datasets.contact_pose_3d import ContactPose3D, collate_fn
import train_test_splits
from utils.misc import setup_logging

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
import visdom
from torch.utils.data import DataLoader
from torch import optim as toptim
from torch import nn as tnn
import numpy as np
import os
import torch
import configparser
import argparse
import logging

import pdb

osp = os.path

def create_plot_window(vis, xlabel, ylabel, title, win, env, trace_name):
  if not isinstance(trace_name, list):
    trace_name = [trace_name]

  vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=trace_name[0],
    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
  for name in trace_name[1:]:
    vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=name)


def train(data_dir, contactpose_dir, split, config_file, experiment_suffix=None,
    checkpoint_dir='.', device_id=0, weights_filename=None, resume_optim=False):
  model_name = config_file.split('/')[-1].split('.')[0]


  # config
  config = configparser.ConfigParser()
  config.read(config_file)

  section = config['optim']
  batch_size = section.getint('batch_size')
  max_epochs = section.getint('max_epochs')
  val_interval = section.getint('val_interval')
  do_val = val_interval > 0
  base_lr = section.getfloat('base_lr')
  momentum = section.getfloat('momentum')
  weight_decay = section.getfloat('weight_decay')

  section = config['misc']
  log_interval = section.getint('log_interval')
  shuffle = section.getboolean('shuffle')
  num_workers = section.getint('num_workers')
  visdom_server = section.get('visdom_server', 'http://localhost')

  section = config['hyperparams']
  droprate = section.getfloat('droprate')
  joint_droprate = section.getfloat('joint_droprate')
  lr_step_size = section.getint('lr_step_size', 10000)
  lr_gamma = section.getfloat('lr_gamma', 1.0)
  pos_weight = section.getfloat('pos_weight')
  n_rotations = section.getint('n_rotations')
  grid_size = section.getint('grid_size')
  uniform_texture_weights = section.getboolean('uniform_texture_weights', False)
  n_surface_features = section.getint('n_feats')

  # --- Loss hyperparameters (paper-consistent) ---
  # Paper: L_all = L_S + lambda_c * L_C, with L_S = L_WCE + L_FL.
  # We keep defaults so existing configs remain valid.
  lambda_c = section.getfloat('lambda_c', 0.1)
  focal_gamma = section.getfloat('focal_gamma', 2.0)
  # If you wish to use class-wise alpha_c in focal loss, provide a comma-separated
  # list in config as: focal_alpha = 1,1,1,... (length = #classes). Default: all 1.
  focal_alpha_str = section.get('focal_alpha', '')
  # Scalar fallback for focal alpha (used when focal_alpha is not provided as a list).
  focal_alpha_scalar = 1.0
  # Which class index represents the "highest-saliency" region for the centroid loss.
  # Default: use the last class.
  centroid_class = section.getint('centroid_class', -1)

  # cuda
  device = 'cuda:{:s}'.format(device_id)

  exp_name = '{:s}_split_{:s}_mesh'.format(model_name, split)
  if experiment_suffix:
    exp_name += '_{:s}'.format(experiment_suffix)
  checkpoint_dir = osp.join(checkpoint_dir, exp_name)
  
  # logging
  if not osp.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  log_filename = osp.join(checkpoint_dir, 'training_log.txt')
  setup_logging(log_filename)
  logger = logging.getLogger()
  logger.info('Config from {:s}:'.format(config_file))
  with open(config_file, 'r') as f:
    for line in f:
      logger.info(line.strip())
  
  # create dataset, loss function and model
  texture_weights = np.load('data/texture_bin_weights.npy')
  texture_weights = torch.from_numpy(texture_weights).to(dtype=torch.float)
  if uniform_texture_weights:
    texture_weights = torch.ones(len(texture_weights))
  logger.info('Texture weights = %s' % texture_weights)
  splits = getattr(train_test_splits, 'split_{:s}'.format(split))
  if 'pointnet_A' in model_name:
    model = PointNetPP_A(n_surface_features, len(texture_weights), droprate)
  elif 'pointnet' in model_name:
    model = PointNetPP(n_surface_features, len(texture_weights), droprate)
  elif 'mlp' in model_name:
    n_hidden_nodes = config['hyperparams'].getint('n_hidden_nodes')
    model = MLPModel(n_surface_features, len(texture_weights), n_hidden_nodes,
        droprate)
  elif 'pointkan_A' in model_name:
    model = PointKAN_A(n_surface_features, len(texture_weights), droprate)
  elif 'pointkan' in model_name:
    model = PointKAN(n_surface_features, len(texture_weights), droprate)
  else:
    raise NotImplementedError

  # Loss components (paper-consistent)
  # L_S = L_WCE + L_FL (softmax multi-class), and L_all = L_S + lambda_c * L_C.
  # We implement omega_t as a sample/point-level reweighting factor.
  ce_loss = tnn.CrossEntropyLoss(ignore_index=-1, reduction='none')
  if do_val:
    val_ce_loss = tnn.CrossEntropyLoss(ignore_index=-1, reduction='none')
  train_dset = ContactPose3D(data_dir, contactpose_dir, grid_size, n_rotations,
      joint_droprate, **splits['train'], train_mode=True)
  if do_val:
    val_dset = ContactPose3D(data_dir, contactpose_dir, grid_size, n_rotations, 0,
                              **splits['test'], train_mode=False)
  # resume model
  if weights_filename is not None:
    checkpoint = torch.load(osp.expanduser(weights_filename), map_location='cpu')
    model.load_state_dict(checkpoint.state_dict(), strict=True)
    logger.info('Loaded weights from {:s}'.format(weights_filename))
  model.to(device=device)
  # move loss helpers / weights to device
  ce_loss.to(device=device)
  texture_weights = texture_weights.to(device=device)
  if do_val:
    val_ce_loss.to(device=device)

  # Parse focal alpha (optional)
  focal_alpha_tensor = None
  if isinstance(focal_alpha_str, str) and len(focal_alpha_str.strip()) > 0:
    try:
      alpha_list = [float(x) for x in focal_alpha_str.split(',')]
      focal_alpha_tensor = torch.tensor(alpha_list, dtype=torch.float, device=device)
    except Exception as e:
      logger.warning('Failed to parse focal_alpha from config ("%s"): %s. Using alpha=1.',
                     focal_alpha_str, str(e))

  # optimizer
  # optim = toptim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay,
  #     momentum=momentum)
  optim = toptim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
  if isinstance(optim, toptim.Adam):
    lr_step_size = 1e10
    logger.info('Optimizer is Adam, disabling LR scheduler')
  lr_scheduler = toptim.lr_scheduler.StepLR(optim, step_size=lr_step_size,
      gamma=lr_gamma)
  
  # resume optim
  if (weights_filename is not None) and resume_optim:
    optim_filename = weights_filename.replace('model', 'optim')
    if osp.isfile(optim_filename):
      checkpoint = torch.load(optim_filename, map_location='cpu')
      optim.load_state_dict(checkpoint.state_dict())
      logger.info('Loaded optimizer from {:s}'.format(optim_filename))

  # dataloader
  train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=shuffle,
    pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
  if do_val:
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=shuffle,
      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

  # checkpointing
  def checkpoint_fn(engine: Engine):
    return -engine.state.avg_loss

  checkpoint_kwargs = dict(dirname=checkpoint_dir, filename_prefix='checkpoint',
    score_function=checkpoint_fn, create_dir=True,  require_empty=False,
      save_as_state_dict=True)
  checkpoint_dict = {'model': model, 'optim': optim}

  # -------------------------
  # Paper-consistent losses
  # -------------------------
  def _compute_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                          alpha: torch.Tensor = None, gamma: float = 2.0,
                          ignore_index: int = -1) -> torch.Tensor:
    """Softmax multi-class focal loss computed only on the ground-truth class."""
    # logits: (P, C), targets: (P,)
    valid = targets.ne(ignore_index)
    if valid.sum().item() == 0:
      return torch.tensor(0.0, device=logits.device)
    t = targets[valid].long()
    log_probs = torch.log_softmax(logits[valid], dim=1)
    probs = torch.exp(log_probs)
    pt = probs.gather(1, t.unsqueeze(1)).squeeze(1)
    logpt = log_probs.gather(1, t.unsqueeze(1)).squeeze(1)
    if alpha is not None:
      a = alpha.gather(0, t).to(dtype=logits.dtype)
    else:
      a = 1.0
    fl = -a * torch.pow(1.0 - pt, gamma) * logpt
    return fl.mean()

  def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.clamp(weights.sum(), min=eps)
    return (values * weights).sum() / denom

  def compute_losses(logits: torch.Tensor,
                     labels: torch.Tensor,
                     coords: torch.Tensor,
                     batch_ids: torch.Tensor = None,
                     texture_ids: torch.Tensor = None):
    """Returns L_all and a dict of components.

    - L_S = L_WCE + L_FL (softmax multi-class)
    - L_all = L_S + lambda_c * L_C
    - omega_t is applied as a point/sample reweighting factor.
    """
    C = logits.shape[1]
    # texture interval id per point/sample
    if texture_ids is None:
      texture_ids = labels
    texture_ids = texture_ids.long()

    valid = labels.ne(-1)
    if valid.sum().item() == 0:
      zero = torch.tensor(0.0, device=logits.device)
      return zero, {'L_WCE': zero, 'L_FL': zero, 'L_S': zero, 'L_C': zero}

    # omega_t (point-wise)
    omega = torch.ones_like(labels, dtype=torch.float, device=logits.device)
    # for valid texture ids, lookup weight; clamp indices for safety
    tex_valid = texture_ids.ne(-1) & valid
    if tex_valid.sum().item() > 0:
      tex_idx = texture_ids[tex_valid].clamp(min=0, max=texture_weights.numel()-1)
      omega[tex_valid] = texture_weights.gather(0, tex_idx)
    omega = omega[valid]

    # ---- Saliency loss: WCE + Focal (both on softmax logits) ----
    ce_per = ce_loss(logits, labels.long())  # (P,)
    ce_per = ce_per[valid]
    log_probs = torch.log_softmax(logits[valid], dim=1)
    probs = torch.exp(log_probs)
    t = labels[valid].long()
    pt = probs.gather(1, t.unsqueeze(1)).squeeze(1)
    logpt = log_probs.gather(1, t.unsqueeze(1)).squeeze(1)
    if focal_alpha_tensor is not None and focal_alpha_tensor.numel() == C:
      a = focal_alpha_tensor.gather(0, t).to(dtype=logits.dtype)
    else:
      a = focal_alpha_scalar
    fl_per = -a * torch.pow(1.0 - pt, focal_gamma) * logpt

    l_wce = _weighted_mean(ce_per, omega)
    l_fl = _weighted_mean(fl_per, omega)
    l_s = l_wce + l_fl

    # ---- Centroid loss (regularizer) ----
    # Encourage grasping around the object centroid via a probability-weighted centroid.
    cls_idx = centroid_class if centroid_class >= 0 else (C - 1)
    cls_idx = int(cls_idx)
    p_high = probs[:, cls_idx].detach()  # (P_valid,)
    coords_v = coords[valid]
    if batch_ids is not None:
      b = batch_ids[valid].long()
      unique_b = torch.unique(b)
    else:
      b = None
      unique_b = torch.tensor([0], device=logits.device)

    lc_list = []
    w_list = []
    for bid in unique_b:
      if b is None:
        sel = torch.ones_like(p_high, dtype=torch.bool)
      else:
        sel = b.eq(bid)
      if sel.sum().item() < 1:
        continue
      xyz = coords_v[sel]
      # gt centroid = mean of all points in the sample
      c_gt = xyz.mean(dim=0)
      w = p_high[sel]
      w_sum = torch.clamp(w.sum(), min=1e-12)
      c_pred = (xyz * w.unsqueeze(1)).sum(dim=0) / w_sum
      lc = torch.sum((c_pred - c_gt) ** 2)
      lc_list.append(lc)
      # sample-level omega: mean omega over points
      w_list.append(omega[sel].mean())
    if len(lc_list) == 0:
      l_c = torch.tensor(0.0, device=logits.device)
    else:
      lc_stack = torch.stack(lc_list)
      w_stack = torch.stack(w_list)
      l_c = _weighted_mean(lc_stack, w_stack)

    l_all = l_s + lambda_c * l_c
    return l_all, {'L_WCE': l_wce, 'L_FL': l_fl, 'L_S': l_s, 'L_C': l_c}

  # train and val loops
  def train_loop(engine: Engine, batch):
    # occs, sdata, ijks, batch, colors = batch
    model.train()
    optim.zero_grad()
    
    if 'pointnet_A' in model_name:
      use_idx = range(1, len(batch))
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      batch[2] = batch[2] / grid_size - 0.5
      pred = model(*batch[1:4])
      texture_ids = batch[5] if len(batch) > 5 else None
      loss, comps = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)
    elif 'mlp' in model_name:
      use_idx = [1, 4]
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      pred = model(batch[1])
      texture_ids = batch[5] if len(batch) > 5 else None
      # MLP branch may not provide coordinates/batch ids; fall back to None.
      coords = batch[2] if len(batch) > 2 else None
      batch_ids = batch[3] if len(batch) > 3 else None
      loss, comps = compute_losses(pred, batch[4], coords=coords if coords is not None else batch[1].new_zeros((pred.shape[0], 3)),
                                  batch_ids=batch_ids, texture_ids=texture_ids)
    elif 'pointnet' in model_name:
      use_idx = range(1, len(batch))
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      batch[2] = batch[2] / grid_size - 0.5
      pred = model(*batch[1:4])
      texture_ids = batch[5] if len(batch) > 5 else None
      loss, comps = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)
    elif 'pointkan_A' in model_name:
      use_idx = range(1, len(batch))
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      batch[2] = batch[2] / grid_size - 0.5
      pred = model(*batch[1:4])
      texture_ids = batch[5] if len(batch) > 5 else None
      loss, comps = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)
    elif 'pointkan' in model_name:
      use_idx = range(1, len(batch))
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      batch[2] = batch[2] / grid_size - 0.5
      pred = model(*batch[1:4])
      texture_ids = batch[5] if len(batch) > 5 else None
      loss, comps = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)

    loss.backward()
    optim.step()
    engine.state.train_loss = loss.item()
    # Expose components for logging/debugging
    engine.state.loss_wce = comps['L_WCE'].item() if 'L_WCE' in comps else None
    engine.state.loss_fl = comps['L_FL'].item() if 'L_FL' in comps else None
    engine.state.loss_c = comps['L_C'].item() if 'L_C' in comps else None
    return loss.item()
  trainer = Engine(train_loop)
  train_checkpoint_handler = ModelCheckpoint(score_name='train_loss',
      **checkpoint_kwargs)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, train_checkpoint_handler,
    checkpoint_dict)

  if do_val:
    def val_loop(engine: Engine, batch):
      # occs, sdata, ijks, batch, colors = batch
      model.eval()
      if 'pointnet' in model_name:
        use_idx = range(1, len(batch))
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        batch[2] = batch[2] / grid_size - 0.5
        with torch.no_grad():
          pred = model(*batch[1:4])
          texture_ids = batch[5] if len(batch) > 5 else None
          loss, _ = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)
      elif 'mlp' in model_name:
        use_idx = [1, 4]
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        with torch.no_grad():
          pred = model(batch[1])
          texture_ids = batch[5] if len(batch) > 5 else None
          coords = batch[2] if len(batch) > 2 else batch[1].new_zeros((pred.shape[0], 3))
          batch_ids = batch[3] if len(batch) > 3 else None
          loss, _ = compute_losses(pred, batch[4], coords=coords, batch_ids=batch_ids, texture_ids=texture_ids)
      elif 'pointnet_A' in model_name:
        use_idx = range(1, len(batch))
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        batch[2] = batch[2] / grid_size - 0.5
        with torch.no_grad():
          pred = model(*batch[1:4])
          texture_ids = batch[5] if len(batch) > 5 else None
          loss, _ = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)
      elif 'pointkan_A' in model_name:
        use_idx = range(1, len(batch))
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        batch[2] = batch[2] / grid_size - 0.5
        with torch.no_grad():
          pred = model(*batch[1:4])
          texture_ids = batch[5] if len(batch) > 5 else None
          loss, _ = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)
      elif 'pointkan' in model_name:
        use_idx = range(1, len(batch))
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        batch[2] = batch[2] / grid_size - 0.5
        with torch.no_grad():
          pred = model(*batch[1:4])
          texture_ids = batch[5] if len(batch) > 5 else None
          loss, _ = compute_losses(pred, batch[4], coords=batch[2], batch_ids=batch[3], texture_ids=texture_ids)


      engine.state.val_loss = loss.item()
      return loss.item()
    valer = Engine(val_loop)
    val_checkpoint_handler = ModelCheckpoint(score_name='val_loss',
        **checkpoint_kwargs)
    valer.add_event_handler(Events.EPOCH_COMPLETED, val_checkpoint_handler,
      checkpoint_dict)

  # callbacks
  vis = visdom.Visdom(server=visdom_server)
  logger.info('Visdom at {:s}'.format(visdom_server))
  loss_win = 'loss'
  create_plot_window(vis, '#Epochs', 'Loss', 'Training and Validation Loss',
    win=loss_win, env=exp_name, trace_name=['train_loss', 'val_loss'])

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    it = (engine.state.iteration - 1) % len(train_dloader)
    engine.state.avg_loss = (engine.state.avg_loss*it + engine.state.output) / \
                            (it + 1)

    if it % log_interval == 0:
      logger.info("{:s} train Epoch[{:03d}/{:03d}] Iteration[{:04d}/{:04d}] "
          "Loss: {:02.4f} lr: {:.4f}".
        format(exp_name, engine.state.epoch, max_epochs, it+1, len(train_dloader),
        engine.state.output, lr_scheduler.get_lr()[0]))
      epoch = engine.state.epoch - 1 +\
              float(it+1)/len(train_dloader)

      vis.line(X=np.array([epoch]), Y=np.array([engine.state.output]),
        update='append', win=loss_win, env=exp_name, name='train_loss')

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_avg_train_loss(engine):
    logger.info('{:s} Epoch[{:03d}/{:03d}] Avg. Training Loss: {:02.4f}'.format(
        exp_name, engine.state.epoch, max_epochs, engine.state.avg_loss))

  if do_val:
    @valer.on(Events.ITERATION_COMPLETED)
    def avg_loss_callback(engine: Engine):
      it = (engine.state.iteration - 1) % len(train_dloader)
      engine.state.avg_loss = (engine.state.avg_loss*it + engine.state.output) / \
                              (it + 1)
      if it % log_interval == 0:
        logger.info("{:s} val Iteration[{:04d}/{:04d}] Loss: {:02.4f}"
          .format(exp_name, it+1, len(val_dloader), engine.state.output))

    @valer.on(Events.EPOCH_COMPLETED)
    def log_val_loss(engine: Engine):
      vis.line(X=np.array([trainer.state.epoch]),
        Y=np.array([engine.state.avg_loss]), update='append', win=loss_win,
        env=exp_name, name='val_loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_val(engine: Engine):
      vis.save([exp_name])
      if val_interval < 0:  # don't do validation
        return
      if engine.state.epoch % val_interval != 0:
        return
      valer.run(val_dloader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def step_lr_scheduler(engine: Engine):
      lr_scheduler.step()

  def reset_avg_loss(engine: Engine):
    engine.state.avg_loss = 0
  trainer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)
  if do_val:
    valer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)

  # Ignite the torch!
  trainer.run(train_dloader, max_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default=osp.join('data', 'mesh_prediction_data'))
  parser.add_argument('--contactpose_dir', default=osp.join('data', 'contactpose_data'))
  parser.add_argument('--checkpoint_dir',
                      default=osp.join('data', 'checkpoints'))
  parser.add_argument('--split', type=str, required=True,
                      choices=('objects', 'participants', 'overfit'))
  parser.add_argument('--config_file', required=True)
  parser.add_argument('--weights_file', default=None)
  parser.add_argument('--suffix', default=None)
  parser.add_argument('--device_id', default='0')
  parser.add_argument('--resume_optim', action='store_true')
  args = parser.parse_args()


  train(args.data_dir, args.contactpose_dir, args.split, args.config_file,
    experiment_suffix=args.suffix, device_id=args.device_id,
    checkpoint_dir=osp.expanduser(args.checkpoint_dir),
    weights_filename=args.weights_file, resume_optim=args.resume_optim)
