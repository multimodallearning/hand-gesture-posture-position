import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from configs.defaults import get_cfg_defaults
from data.shrec17_dataset import SHREC17Dataset
from model.model import TwoStreamLSTM


def test(cfg, model_path):

    # prepare test dataset
    test_set = SHREC17Dataset(cfg, phase='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    # MODEL
    model = TwoStreamLSTM(cfg).to(cfg.MODEL.DEVICE)
    model.load_state_dict(torch.load(model_path))

    # for inference, we only use the last hidden state from LSTM for classification
    model.num_ts_per_pred = 1
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#model parameters: ', num_params)

    # Start evaluation
    model.eval()
    num_correct = 0
    # confusion matrix: [[TN, FP], ---> upper row: actually negative, lower row: actually positive
    #                    [FN, TP]]      left column: predicted negative, right column: predicted positive
    cm = np.zeros([cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES])

    # prepare for measurement of inference times
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_inference_times = []

    for it, data in enumerate(test_loader, 1):
        inputs, labels, _ = data
        inputs = [inp.to(cfg.MODEL.DEVICE) for inp in inputs]
        labels = labels.to(cfg.MODEL.DEVICE)

        with torch.no_grad():
            torch.cuda.synchronize()
            starter.record()
            pred = model(inputs)

        pred = pred.argmax(dim=1)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        all_inference_times.append(curr_time)
        num_correct += torch.sum(pred.view(-1) == labels.view(-1)).item()

        for i, j in zip(labels.view(-1), pred):
            cm[i, j] += 1

    test_accuracy = num_correct / len(test_set)

    print('average inference time: ', np.mean(all_inference_times))
    print('accuracy: ', test_accuracy)


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
    parser.add_argument(
        "--model-path",
        default="",
        metavar="FILE",
        help="path to pre-trained model",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.model_path == "":
        working_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
        model_path = os.path.join(working_directory, 'model.pth')
    else:
        model_path = args.model_path
    if not os.path.isfile(model_path):
        raise ValueError('There is no pre-trained model at the specified path. Set the model path correctly or'
                         ' run training first.')

    test(cfg, model_path)
