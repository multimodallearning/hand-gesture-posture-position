import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones.bps_densenet import BPSDensenet
from model.backbones.cnn3d import VoxNet
from model.backbones.dgcnn import DGCNN_cls
from model.backbones.pointnet import PointNetCls
from model.backbones.pointnet2 import PointNet2


class TwoStreamLSTM(nn.Module):

    def __init__(self, cfg):
        super(TwoStreamLSTM, self).__init__()
        self.fusion_type = cfg.MODEL.FUSION_TYPE
        self.input_type = cfg.MODEL.INPUT_COMBINATION

        self.num_ts_per_pred = cfg.MODEL.NUM_TIMESTEPS_PER_PRED
        self.local_scaler = cfg.MODEL.LOCAL_SCALER
        self.global_scaler = cfg.MODEL.GLOBAL_SCALER
        num_classes = cfg.MODEL.NUM_CLASSES

        local_embed = 512
        if cfg.MODEL.LOCAL_BACKBONE == 'pointnet':
            self.local_backbone = PointNetCls(embed=local_embed)
        elif cfg.MODEL.LOCAL_BACKBONE == 'dgcnn':
            self.local_backbone = DGCNN_cls(cfg, final_embed=local_embed)
        elif cfg.MODEL.LOCAL_BACKBONE == 'pointnet2':
            self.local_backbone = PointNet2(embed=local_embed)
        elif cfg.MODEL.LOCAL_BACKBONE == 'bps':
            self.local_backbone = BPSDensenet(cfg, local_embed)
        else:
            raise ValueError

        global_embed = 1024
        if cfg.MODEL.GLOBAL_BACKBONE == 'pointnet':
            self.global_backbone = PointNetCls(embed=global_embed)
        elif cfg.MODEL.GLOBAL_BACKBONE == 'dgcnn':
            self.global_backbone = DGCNN_cls(cfg, final_embed=global_embed)
        elif cfg.MODEL.GLOBAL_BACKBONE == 'bps':
            self.global_backbone = BPSDensenet(cfg, global_embed)
        elif cfg.MODEL.GLOBAL_BACKBONE == 'voxnet':
            self.global_backbone = VoxNet(embed=global_embed)
        else:
            raise ValueError()

        if self.fusion_type in ['no_fusion', 'late_fusion', 'intermediate_fusion']:
            self.local_lstm = nn.LSTM(local_embed, 256, 1, batch_first=True)
            self.global_lstm = nn.LSTM(global_embed, 256, 1, batch_first=True)

            if self.fusion_type == 'no_fusion':
                self.local_heads = nn.ModuleList([nn.Dropout(0.5),
                                                  nn.Linear(256, num_classes)])
                self.global_heads = nn.ModuleList([nn.Dropout(0.5),
                                                  nn.Linear(256, num_classes)])
            elif self.fusion_type == 'late_fusion':
                self.fusion_heads = nn.ModuleList([nn.Dropout(0.5),
                                                   nn.Linear(2 * 256, 128),
                                                   nn.ReLU(),
                                                   nn.BatchNorm1d(128),
                                                   nn.Dropout(0.5),
                                                   nn.Linear(128, num_classes)])
            elif self.fusion_type == 'intermediate_fusion':
                self.fusion_lstm = nn.LSTM(2 * 256, 256, 1, batch_first=True)
                self.heads = nn.ModuleList([nn.Dropout(0.5),
                                            nn.Linear(256, num_classes)])
            else:
                raise ValueError()

        elif self.fusion_type == 'early_fusion':
            if self.input_type == 'local_global':
                input_size = local_embed + global_embed
            elif self.input_type == 'local':
                input_size = local_embed
            elif self.input_type == 'global':
                input_size = global_embed
            else:
                raise ValueError()

            self.lstm = nn.LSTM(input_size, 256, 1, batch_first=True)
            self.heads = nn.ModuleList([nn.Dropout(0.5),
                                        nn.Linear(256, num_classes)])

        else:
            raise ValueError()

    def forward(self, x):
        x_loc, x_glob = x
        if self.global_scaler > 0:
            x_glob /= self.global_scaler
        else:
            dists = torch.sqrt(torch.sum(torch.square(x_glob), dim=3))
            dists_per_T = torch.max(dists, dim=2, keepdim=True)[0]
            dists_per_B = torch.max(dists_per_T, dim=1, keepdim=True)[0]
            x_glob /= dists_per_B.unsqueeze(-1)

        if self.local_scaler > 0:
            x_loc /= self.local_scaler
        else:
            dists = torch.sqrt(torch.sum(torch.square(x_loc), dim=3))
            dists_per_T = torch.max(dists, dim=2, keepdim=True)[0]
            x_loc /= dists_per_T.unsqueeze(-1)

        if self.training:
            x_loc = x_loc[:, :, torch.randperm(x_loc.shape[2])[:128], :]
            x_glob = x_glob[:, :, torch.randperm(x_glob.shape[2])[:128], :]
        else:
            x_loc = x_loc[:, :, ::x_loc.shape[2] // 128, :]
            x_glob = x_glob[:, :, ::x_glob.shape[2] // 128, :]

        # GLOBAL ENCODING
        if 'global' in self.input_type:
            B, T, N, D = x_glob.size()
            x_glob = x_glob.view(-1, N, D)
            x_glob = x_glob.transpose(1, 2)
            x_glob = self.global_backbone(x_glob)
            x_glob = x_glob.view(B, T, -1)

        # LOCAL ENCODING
        if 'local' in self.input_type:
            B, T, N, D = x_loc.size()
            x_loc = x_loc.view(-1, N, D)
            x_loc = x_loc.transpose(1, 2)
            x_loc = self.local_backbone(x_loc)
            x_loc = x_loc.view(B, T, -1)

        if self.fusion_type == 'early_fusion':
            if self.input_type == 'local_global':
                x = torch.cat((x_glob, x_loc), dim=2)
            elif self.input_type == 'local':
                x = x_loc
            elif self.input_type == 'global':
                x = x_glob

            out, (ht, ct) = self.lstm(x)
            if self.num_ts_per_pred == 1:
                x = ht[-1]
            elif self.num_ts_per_pred > 1:
                x = out[:, -self.num_ts_per_pred:, :]
                x = x.reshape(-1, x.size(2))
            for layer in self.heads:
                x = layer(x)
            return x

        else:
            out_glob, (ht_glob, ct_glob) = self.global_lstm(x_glob)
            out_loc, (ht_loc, ct_loc) = self.local_lstm(x_loc)

            if self.fusion_type == 'intermediate_fusion':
                x = torch.cat((out_glob, out_loc), dim=2)
                out, (ht, ct) = self.fusion_lstm(x)

                if self.num_ts_per_pred == 1:
                    x = ht[-1]
                elif self.num_ts_per_pred > 1:
                    x = out[:, -self.num_ts_per_pred:, :]
                    x = x.reshape(-1, x.size(2))
                for layer in self.heads:
                    x = layer(x)
                return x

            else:
                if self.num_ts_per_pred == 1:
                    x_glob = ht_glob[-1]
                    x_loc = ht_loc[-1]
                elif self.num_ts_per_pred > 1:
                    x_glob = out_glob[:, -self.num_ts_per_pred:, :]
                    x_glob = x_glob.reshape(-1, x_glob.size(2))
                    x_loc = out_loc[:, -self.num_ts_per_pred:, :]
                    x_loc = x_loc.reshape(-1, x_loc.size(2))

                if self.fusion_type == 'late_fusion':
                    x = torch.cat((x_loc, x_glob), dim=1)
                    for layer in self.fusion_heads:
                        x = layer(x)
                    return x
                elif self.fusion_type == 'no_fusion':
                    for layer in self.global_heads:
                        x_glob = layer(x_glob)
                    for layer in self.local_heads:
                        x_loc = layer(x_loc)
                    x = (F.log_softmax(x_glob, dim=1) + F.log_softmax(x_loc, dim=1)) / 2
                    return x
