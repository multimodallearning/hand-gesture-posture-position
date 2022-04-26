### adapted from https://github.com/ycmin95/pointlstm-gesture-recognition-pytorch/blob/master/dataset/shrec17_process.py

import sys
sys.path.append('../')
import re
import os
import cv2
import copy
import imageio
import numpy as np
import data.process_utils as utils


if __name__ == "__main__":
    pts_size = 512
    use_dbscan = True

    r = re.compile('[ \t\n\r]+')
    dataset_prefix = "../dataset/DHG/raw"
    prefix = dataset_prefix + "/gesture_{}/finger_{}/subject_{}/essai_{}"
    input_list = open(dataset_prefix + "/informations_troncage_sequences.txt").readlines()
    for idx, line in enumerate(input_list):
        # Loading dataset
        splitLine = r.split(line)
        dir_path = prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
        print(idx, len(input_list), dir_path)
        hand_regions = np.loadtxt(dir_path + '/general_information.txt').astype(int)
        pts = np.zeros((32, pts_size, 8), dtype=int)
        start_frame = int(splitLine[4])
        stop_frame = int(splitLine[5])
        duration = stop_frame - start_frame + 1
        ind = utils.key_frame_sampling(duration, 32)
        real_ind = np.arange(start_frame, stop_frame+1, 1)
        ind = real_ind[ind]
        hand_regions = hand_regions[ind]
        depth_video = []
        for i, frame_id in enumerate(ind):
            # Reconstruct point cloud sequence from depth video
            depth_image = imageio.imread(dir_path + "/depth_{}.png".format(frame_id + 1))
            depth_video.append(depth_image)
            hand_crop = depth_image[hand_regions[i][2]:hand_regions[i][2] + hand_regions[i][4],
                        hand_regions[i][1]:hand_regions[i][1] + hand_regions[i][3]]
            hand_crop = cv2.medianBlur(hand_crop, 3)
            pts[i, :, :4] = utils.generate_pts_cloud_sequence(hand_crop, hand_regions, pts_size, i, use_dbscan)
            pts[i, :, 4:8] = utils.uvd2xyz_shrec(copy.deepcopy(pts[i, :, :4]))
        save_dir = dir_path.replace('raw', 'processed/dbscanCluster_numPts=512')
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
        np.save(save_dir + "/pts_label.npy", pts)
