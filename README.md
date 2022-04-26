# hand-gesture-posture-position
Source code for our 3DV 2021 paper [Fusing Posture and Position Representations for Point Cloud-based Hand Gesture Recognition](https://ieeexplore.ieee.org/abstract/document/9665889) [[pdf](https://csdl-downloads.ieeecomputer.org/proceedings/3dv/2021/2688/00/268800a617.pdf?Expires=1650981378&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jc2RsLWRvd25sb2Fkcy5pZWVlY29tcHV0ZXIub3JnL3Byb2NlZWRpbmdzLzNkdi8yMDIxLzI2ODgvMDAvMjY4ODAwYTYxNy5wZGYiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NTA5ODEzNzh9fX1dfQ__&Signature=dGJAowaa~BXFUu5aFDd6gNVANrL4o8Fe-bkfwwDriNtftsYpSr0AoBqXQDT0oP16Qth-Obl2X8plKTRoAYlzIhOcUGcTBbs3V0~c2SlCVez4AfxFscMBio0RkshpK89GoBPHO~Ltd1zImCIjVz9tLD1Js9naFpreYbpH26M8zK8arWsmXQ7GaZ0p2bxls2Y2~CmtVGkNSx7qnGRX0sL0ikqMhf9eYZtvzztJld0~scixI4Om9OgOzDE8SXFFuxaTcJppIrKCVHQQ9wIPxsGqjtjqvBR3JDLuMKWCDCW-nSeQybmtL4VyEslqaiBIVHOdkw4M~Klb9dVxRdCIzRQRjQ__&Key-Pair-Id=K12PMWTCQBDMDT)].

## Dependencies
Please first install the following dependencies
* Python3 (we use 3.8.3)
* numpy
* pytorch (we use 1.6.0)
* bps
* yacs
* cv2
* sklearn
* scikit-image
* imageio
* pytorch-geometric (only for ablation experiment with PointNet++)

## Data Preparation
### Shrec
1. Download the Shrec'17 dataset from http://www-rech.telecom-lille.fr/shrec2017-hand/. Create a directory `/dataset/shrec17` and move `HandGestureDataset_SHREC2017` to this directory. We recommend to create a symlink.
2. Create a directory `/dataset/shrec17/Processed_HandGestureDataset_SHREC2017`. Subsequently, execute `cd data` and `python shrec17_process.py`to generate point cloud sequences from the original depth images.

### DHG
1. Download the DHG dataset from http://www-rech.telecom-lille.fr/DHGdataset/. Create a directory `/dataset/DHG/raw` and move the extracted dataset to this directory. We recommend to create a symlink.
2. Create a directory `/dataset/DHG/processed`. Subsequently, execute `cd data` and `python dhg_process.py`to generate point cloud sequences from the original depth images.


## Training
### Shrec
1. In `/configs/defaults.py`, modify `_C.BASE_DIRECTORY` in line 5 to the root directory where you intend to save the results.
2. In the config files `/configs/config_full-model_shrec*.yaml`, you can optionally modify `EXPERIMENT_NAME`in line 1. Models and log files will finally be written to `os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)`.
3. Navigate to the `main` directory and execute `python train.py --config-file "../configs/config_full-model_shrec*.yaml" --gpu GPU` to train our full model on the Shrec'17 dataset. * should be either 14 or 28 depending on the protocol you want to train on.
4. After each epoch, we save the model weights and a log file to the specified directory.

### DHG
The procedure is analogous to the Shrec dataset.
Just navigate to the `main` directory and execute `python train_dhg.py --config-file "../configs/config_full-model_dhg28.yaml" --gpu GPU` to train our full model on the DHG28 dataset.
Note that we perform leave-one-fold-out cross-validation, i.e. this will run 20 successive trainings.

## Testing
### Shrec
* If you trained a model yourself following the instructions above, you can test the model by executing `python test.py --config-file "../configs/config_full-model_shrec*.yaml" --gpu GPU`. The output comprises recognition accuracy, number of model parameters and inference time averaged over all batches.
* Otherwise, we provide pre-trained models for [Shrec 14G](https://drive.google.com/file/d/1JIpOjM36upTdm-MCvjuOZNnuWKpvKdf3/view?usp=sharing) protocol and for [Shrec 28G](https://drive.google.com/file/d/195_gpv8LYQsdYtPVDMsSFntLyjuZKtjt/view?usp=sharing) protocol. Download the models and use them for inference by executing `python test.py --config-file "../configs/config_full-model_shrec*.yaml" --gpu GPU --model-path PATH/TO/MODEL`. The provided models achieve 95.24% under the 28G protocol and 96.43% under the 14G protocol.

### DHG
Again, the procedure is analogous to the Shrec dataset.
Once you finished training on the DHG dataset, you can test all models on the associated data split by executing `python test.py --config-file "../configs/config_full-model_dhg28.yaml" --gpu GPU`

## Citation
If you find our code useful for your work, please cite the following paper
```latex
@inproceedings{bigalke2021fusing,
  title={Fusing Posture and Position Representations for Point Cloud-Based Hand Gesture Recognition},
  author={Bigalke, Alexander and Heinrich, Mattias P},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={617--626},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgements
* Code for data pre-processing has been adapted from https://github.com/ycmin95/pointlstm-gesture-recognition-pytorch/tree/master/dataset
* DGCNN implementation has been adapted from https://github.com/AnTao97/dgcnn.pytorch
* PointNet++ implementation has been adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

We thank all authors for sharing their code!
