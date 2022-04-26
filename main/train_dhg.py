import sys
sys.path.append('../')
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.distributions.normal import Normal
from configs.defaults import get_cfg_defaults
from data.dhg_dataset import DHGDataset
from model.model import TwoStreamLSTM

def train(cfg, output_directory):
    val_subject_ids = cfg.DATASETS.CV_SUBJECT_IDS
    num_splits = len(val_subject_ids)
    # logging: save training loss, valiadation loss and validation accuracy after each epoch to numpy array
    validation_log = np.zeros([num_splits, cfg.SOLVER.NUM_EPOCHS, 3])

    for k, val_subjects in enumerate(val_subject_ids):
        print("Split {}/{}: validate on subjects {}".format(k+1, num_splits, val_subjects))

        # prepare datasets
        num_workers = cfg.INPUT.NUM_WORKERS
        train_set = DHGDataset(cfg, phase='train', val_subjects=val_subjects)
        train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN, num_workers=num_workers, shuffle=True,drop_last=True)
        val_set = DHGDataset(cfg, phase='test', val_subjects=val_subjects)
        val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=num_workers, shuffle=False, drop_last=False)

        # model
        model = TwoStreamLSTM(cfg).to(cfg.MODEL.DEVICE)

        # optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)
        weight_noise = cfg.SOLVER.WEIGHT_NOISE

        # loss criterion
        if cfg.SOLVER.LOSS_FUNCTION == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(reduction="mean")
        # use NLL loss for 'no fusion' in ablation experiment
        elif cfg.SOLVER.LOSS_FUNCTION == 'nll':
            criterion = nn.NLLLoss()
        else:
            raise ValueError('The specified loss function {} is not available'.format(cfg.SOLVER.LOSS_FUNCTION))

        # Training
        for e in range(cfg.SOLVER.NUM_EPOCHS):
            start_time = time.time()
            model.train()
            loss_values = []
            for it, data in enumerate(train_loader, 1):
                inputs, labels, _ = data
                # inputs comprises
                inputs = [inp.to(cfg.MODEL.DEVICE) for inp in inputs]
                labels = labels.to(cfg.MODEL.DEVICE)

                # apply weight noise to LSTM parameters
                if weight_noise > 0.:
                    orig_params = []
                    for n, p in model.named_parameters():
                        orig_params.append(p.clone())
                        if 'lstm' in n:
                            gaussian = Normal(loc=0, scale=torch.ones_like(p) * weight_noise)
                            p.data = p.data + gaussian.sample()

                pred = model(inputs)

                # in case we perform classification from multiple hidden states of the LSTM, we need to duplicate the labels
                # accordingly
                if model.num_ts_per_pred > 1:
                    labels = torch.repeat_interleave(labels, model.num_ts_per_pred)

                loss = criterion(pred, labels.view(-1))

                optimizer.zero_grad()
                loss.backward()

                # reset LSTM parameters to original weights
                if weight_noise > 0.:
                    for p, orig_p in zip(model.parameters(), orig_params):
                        p.data = orig_p.data

                optimizer.step()

                loss_values.append(loss.item())
                if it % 20 == 0:
                    mean_loss = np.mean(loss_values[-20:])
                    print('Epoch: {}, Batch({}/{}), Loss: {:.4f}'
                            .format(e, it, len(train_loader), mean_loss))

            lr_scheduler.step()

            # Validate after each epoch
            model.eval()

            train_loss = np.mean(loss_values)
            loss_values = []
            num_correct = 0

            for it, data in enumerate(val_loader, 1):
                inputs, labels, _ = data
                inputs = [inp.to(cfg.MODEL.DEVICE) for inp in inputs]
                labels = labels.to(cfg.MODEL.DEVICE)

                with torch.no_grad():
                    pred = model(inputs)

                if model.num_ts_per_pred > 1:
                    labels = torch.repeat_interleave(labels, model.num_ts_per_pred)

                loss = criterion(pred, labels.view(-1))
                loss_values.append(loss.item())

                # in case we perform classification from multiple hidden states of the LSTM, we average the softmax scores
                if model.num_ts_per_pred > 1:
                    pred = pred.reshape(-1, model.num_ts_per_pred, pred.size(1))
                    pred = F.softmax(pred, dim=2)
                    pred = torch.mean(pred, dim=1)
                    labels = labels[::model.num_ts_per_pred]

                pred = pred.argmax(dim=1)
                num_correct += torch.sum(pred.view(-1) == labels.view(-1)).item()

            val_loss = np.mean(loss_values)
            val_acc = num_correct / len(val_set)

            end_time = time.time()

            print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.3f' % train_loss,
                  'val_loss', '%0.3f' % val_loss, 'val_acc', '%0.4f' % val_acc)

            validation_log[k, e, :] = [train_loss, val_loss, val_acc]
            np.save(os.path.join(output_directory, "validation_log.npy"), validation_log)
            torch.save(model.state_dict(), os.path.join(output_directory, 'model_{}.pth'.format(k)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train(cfg, output_directory)
