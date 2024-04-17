import time
import torch

from model.MyNet import MyNet
from model.AlexNet import AlexNet
from model.GoogLeNet import GoogLeNet
from model.LeNet import LeNet
from model.ResNet import ResNet
from model.VGG import VGG
from utils.dataset import get_dataloader
from utils.utils import *


def main(args):
    model_type = args.model_type
    test_datadir = args.test_datadir
    output_path = args.output_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    match model_type:
        case 'AlexNet':
            model = AlexNet()
            model.load_state_dict(
                torch.load('./checkpoint/alexnet_best.pth',
                           map_location=torch.device('cpu')))
        case 'GoogLeNet':
            model = GoogLeNet()
            model.load_state_dict(
                torch.load('./checkpoint/googlenet_best.pth',
                           map_location=torch.device('cpu')))
        case 'LeNet':
            model = LeNet()
            model.load_state_dict(
                torch.load('./checkpoint/lenet_best.pth',
                           map_location=torch.device('cpu')))
        case 'MyNet':
            model = MyNet()
            model.load_state_dict(
                torch.load('./checkpoint/mynet_best.pth',
                           map_location=torch.device('cpu')))
        case 'ResNet':
            model = ResNet()
            model.load_state_dict(
                torch.load('./checkpoint/resnet_best.pth',
                           map_location=torch.device('cpu')))
        case 'VGG':
            model = VGG()
            model.load_state_dict(
                torch.load('./checkpoint/vgg_best.pth',
                           map_location=torch.device('cpu')))
        case _:
            raise NameError('Unknown model type')

    model.to(device)

    test_loader = get_dataloader(test_datadir, batch_size=1, split='test')

    predictions = []
    model.eval()
    with torch.no_grad():
        test_start_time = time.time()
        for data in test_loader:
            data = data['images'].to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.append(pred.item())

    test_time = time.time() - test_start_time
    print()
    print(
        f'Finish testing {test_time:.2f} sec(s), dumps result to {output_path}'
    )

    print("=====start to evaluate======")

    write_csv(output_path, predictions, test_loader)
    pred_files, pred_labels = read_csv(args.csv_path)
    gt_files, gt_labels = read_json(args.annos_path)

    test_correct = 0.0
    for i, filename in enumerate(pred_files):
        if gt_labels[gt_files.index(filename)] == pred_labels[i]:
            test_correct += 1

    test_acc = test_correct / len(pred_files)
    print(f'Accuracy = {test_acc}\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
