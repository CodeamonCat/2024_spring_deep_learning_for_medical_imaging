import argparse
import csv
import json
import numpy as np
import os
import random
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annos_path',
                        help='ground truth json file path',
                        type=str,
                        default='../dataset/val/annotations.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--csv_path',
                        help='predicted csv file path',
                        type=str,
                        default='./output/pred.csv')
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='./dataset/',
                        help='dataset directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='DLMI_')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--milestones', type=list, default=[16, 32, 45])
    parser.add_argument('--model_type', type=str, default='GoogLeNet')
    parser.add_argument('--output_path',
                        help='output csv file path',
                        type=str,
                        default='./checkpoint/pred.csv')
    parser.add_argument('--root', type=str, default=".")
    parser.add_argument('--size',
                        type=int,
                        default=224,
                        help='resized image size')
    parser.add_argument('--test_datadir',
                        help='test dataset directory',
                        type=str,
                        default='../dataset/val/')
    parser.add_argument('--use_adam', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def write_config_log(logfile_path):
    args = get_args()
    with open(logfile_path, 'w') as f:
        f.write(f'Experiment Name = {args.exp_name}\n')
        f.write(f'Model Type      = {args.model_type}\n')
        f.write(f'Num epochs      = {args.epochs}\n')
        f.write(f'Batch size      = {args.batch_size}\n')
        f.write(f'Use adam        = {args.use_adam}\n')
        f.write(f'Learning rate   = {args.lr}\n')
        f.write(f'Scheduler step  = {args.milestones}\n')


def write_result_log(logfile_path, epoch, epoch_time, train_acc, val_acc,
                     train_loss, val_loss, is_better):
    args = get_args()
    with open(logfile_path, 'a') as f:
        f.write(
            f'[{epoch + 1}/{args.epochs}] {epoch_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Val Acc: {val_acc:.5f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}'
        )
        if is_better:
            f.write(' -> val best (acc)')
        f.write('\n')


def write_csv(output_path, predictions, test_loader):
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i, label in enumerate(predictions):
            filename = test_loader.dataset.image_names[i]
            writer.writerow([filename, str(label)])


def read_csv(filepath):
    with open(filepath, 'r', newline='') as f:
        data = csv.reader(f)
        header = next(data)
        data = list(data)
    filenames = [x[0] for x in data]
    labels = [int(x[1]) for x in data]
    return filenames, labels


def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    filenames = data['filenames']
    labels = data['labels']
    return filenames, labels
