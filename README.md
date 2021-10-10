# hand-gesture-posture-position
Source code for our 3DV 2021 paper, entitled "Fusing Posture and Position Representations for Point Cloud-based Hand Gesture Recognition"

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
1. Download the Shrec'17 dataset from http://www-rech.telecom-lille.fr/shrec2017-hand/. Create a directory `/dataset/shrec17` and move `HandGestureDataset_SHREC2017` to this directory. We recommend to create a symlink.
2. Create a directory `/dataset/shrec17/Processed_HandGestureDataset_SHREC2017`. Subsequently, execute `cd data` and `python shrec17_process.py`to generate point cloud sequences from the original depth images.

## Training
1. In `/configs/defaults.py`, modify `_C.BASE_DIRECTORY` in line 5 to the root directory where you intend to save the results.
2. In the config files `/configs/config_full-model_shrec*.yaml`, you can optionally modify `EXPERIMENT_NAME`in line 1. Models and log files will finally be written to `os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)`.
3. Navigate to the `main` directory and execute `python train.py --config-file "../configs/config_full-model_shrec*.yaml" --gpu GPU` to train our full model on the Shrec'17 dataset. * should be either 14 or 28 depending on the protocol you want to train on.
4. After each epoch, we save the model weights and a log file to the specified directory.

## Testing
* If you trained a model yourself following the instructions above, you can test the model by executing `python test.py --config-file "../configs/config_full-model_shrec*.yaml" --gpu GPU`. The output comprises recognition accuracy, number of model parameters and inference time averaged over all batches.
* Otherwise, we provide pre-trained models for [Shrec 14G](https://drive.google.com/file/d/1JIpOjM36upTdm-MCvjuOZNnuWKpvKdf3/view?usp=sharing) protocol and for [Shrec 28G](https://drive.google.com/file/d/195_gpv8LYQsdYtPVDMsSFntLyjuZKtjt/view?usp=sharing) protocol. Download the models and use them for inference by executing `python test.py --config-file "../configs/config_full-model_shrec*.yaml" --gpu GPU --model-path PATH/TO/MODEL`. The provided models achieve 95.24% under the 28G protocol and 96.43% under the 14G protocol.

## Acknowledgements
* Code for data pre-processing has been adapted from https://github.com/ycmin95/pointlstm-gesture-recognition-pytorch/tree/master/dataset
* DGCNN implementation has been adapted from https://github.com/AnTao97/dgcnn.pytorch
* PointNet++ implementation has been adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

We thank all authors for sharing their code!
