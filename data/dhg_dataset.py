import re
import os
import numpy as np
import torch.utils.data


class DHGDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, phase, val_subjects):
        self.phase = phase
        self.num_classes = cfg.DATASETS.DHG.NUM_CLASSES
        self.val_subjects = val_subjects
        self.r = re.compile('[ \t\n\r:]+')
        self.inputs_list = self.get_inputs_list()
        subfolder = cfg.DATASETS.DHG.PROCESS_FOLDER
        self.prefix = "../dataset/DHG/processed/"  + subfolder +  "/gesture_{}/finger_{}/subject_{}/essai_{}"

    def __getitem__(self, index):
        splitLine = self.r.split(self.inputs_list[index])

        # get label
        gesture = int(splitLine[0])
        finger = int(splitLine[1])
        label14 = gesture - 1
        label28 = (gesture - 1) * 2 + (finger - 1)

        input_data = np.float32(np.load(
            self.prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
            + "/pts_label.npy"))[:, :, :7]

        pcd = input_data[:, :, 4:7]

        if self.phase == 'train':
            T, N, D = pcd.shape
            pcd = pcd.reshape(-1, D)
            pcd -= np.mean(pcd, axis=0)
            pcd, rot_angle = self.rotate(pcd)
            pcd, scale_value = self.scale(pcd)
            pcd = pcd.reshape(T, N, D)

        pcd_loc = pcd.copy()
        pcd_glob = pcd.copy()

        # mean-center each frame separately for local stream
        means_per_frame = np.mean(pcd_loc, axis=1)
        pcd_loc -= means_per_frame[:, None, :]

        # mean-center the whole sequence for global stream
        T, N, D = pcd_glob.shape
        pcd_glob = pcd_glob.reshape(-1, D)
        pcd_glob -= np.mean(pcd_glob, axis=0)
        pcd_glob = pcd_glob.reshape(T, N, D)

        input_data = [pcd_loc, pcd_glob]

        if self.num_classes == 28:
            label = label28
        elif self.num_classes == 14:
            label = label14
        else:
            raise ValueError("DHG Dataset only provides 14 or 28 label classes, not {} as specified".format(self.num_classes))

        return input_data, label, self.inputs_list[index]

    def get_inputs_list(self):
        inputs_path = "../dataset/DHG/raw/informations_troncage_sequences.txt"
        inputs_list = open(inputs_path).readlines()
        ret_line = []
        if self.phase == 'train':
            for line in inputs_list:
                splitLine = self.r.split(line)
                subject = int(splitLine[2])
                if subject not in self.val_subjects:
                    ret_line.append((line))
        elif self.phase == 'test':
            for line in inputs_list:
                splitLine = self.r.split(line)
                subject = int(splitLine[2])
                if subject in self.val_subjects:
                    ret_line.append(line)

        return ret_line

    def rotate(self, pcd):
        rot_angle = np.clip(np.random.normal(0., 0.06, 3), -0.18, 0.18)
        cosval = np.cos(rot_angle[0])
        sinval = np.sin(rot_angle[0])
        R_x = np.array([[1., 0., 0.], [0., cosval, -sinval], [0., sinval, cosval]], dtype=np.float32)
        cosval = np.cos(rot_angle[1])
        sinval = np.sin(rot_angle[1])
        R_y = np.array([[cosval, 0., sinval], [0., 1., 0.], [-sinval, 0., cosval]], dtype=np.float32)
        cosval = np.cos(rot_angle[2])
        sinval = np.sin(rot_angle[2])
        R_z = np.array([[cosval, -sinval, 0.], [sinval, cosval, 0.], [0., 0., 1.]], dtype=np.float32)
        rotation_matrix = np.dot(np.dot(R_z, R_y), R_x)
        pcd = np.dot(pcd, rotation_matrix.T)
        return pcd, rot_angle

    def scale(self, pcd):
        scale_value = np.random.uniform(0.9, 1.1)
        pcd *= scale_value
        return pcd, scale_value

    def __len__(self):
        return len(self.inputs_list)
