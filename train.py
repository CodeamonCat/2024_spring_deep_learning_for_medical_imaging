import os
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from model.MyNet import MyNet
from model.AlexNet import AlexNet
from model.GoogLeNet import GoogLeNet
from model.LeNet import LeNet
from model.ResNet import ResNet
from model.VGG import VGG
from utils.dataset import get_dataloader
from utils.utils import *


def plot_learning_curve(logfile_dir, result_lists):

    epoch = range(0, len(result_lists['train_acc']))

    # train_acc_list
    plt.figure(0)
    plt.plot(epoch, result_lists['train_acc'])
    plt.title(f'train_acc_list')
    plt.xlabel('epoch'), plt.ylabel('accuracy')
    plt.savefig(os.path.join(logfile_dir, f'train_acc_list.png'))
    plt.show()
    # train_loss_list
    plt.figure(1)
    plt.plot(epoch, result_lists['train_loss'])
    plt.title(f'train_loss_list')
    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.savefig(os.path.join(logfile_dir, f'train_loss_list.png'))
    plt.show()
    # val_acc_list
    plt.figure(2)
    plt.plot(epoch, result_lists['val_acc'])
    plt.title(f'val_acc_list')
    plt.xlabel('epoch'), plt.ylabel('accuracy')
    plt.savefig(os.path.join(logfile_dir, f'val_acc_list.png'))
    plt.show()
    # val_loss_list
    plt.figure(3)
    plt.plot(epoch, result_lists['val_loss'])
    plt.title(f'val_loss_list')
    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.savefig(os.path.join(logfile_dir, f'val_loss_list.png'))
    plt.show()


def train(model, train_loader, val_loader, logfile_dir, model_save_dir,
          criterion, optimizer, scheduler, device):

    train_loss_list, val_loss_list = list(), list()
    train_acc_list, val_acc_list = list(), list()
    best_acc = 0.0

    for epoch in range(args.epochs):
        # train
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(
                f'\r[{epoch + 1}/{args.epochs}] Train batch: {batch + 1} / {len(train_loader)}'
            )
            sys.stdout.flush()
            images, labels = data['images'].to(device), data['labels'].to(
                device)
            pred = model(images)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()

        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu())
        train_loss_list.append(train_loss)

        print()
        print(
            f'[{epoch + 1}/{args.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}'
        )

        # validation
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0

            for data in val_loader:
                images, labels = data['images'].to(device), data['labels'].to(
                    device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred.eq(labels.view_as(pred)).sum().item())

        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print()
        print(
            f'[{epoch + 1}/{args.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}'
        )

        scheduler.step()

        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir,
                                      'result_log.txt'), epoch, epoch_time,
                         train_acc, val_acc, train_loss, val_loss, is_better)

        if is_better:
            print(
                f'[{epoch + 1}/{args.epochs}] Save best model to {model_save_dir} ...'
            )
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }

        plot_learning_curve(logfile_dir, current_result_lists)


def main(args):
    dataset_dir = args.dataset_dir

    # experiment name and write log file
    exp_name = args.model_type + datetime.now().strftime(
        '_%Y_%m_%d_%H_%M_%S') + '_' + args.exp_name
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    set_seed(1216)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ### model
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)
    match args.model_type:
        case 'AlexNet':
            model = AlexNet()
        case 'GoogLeNet':
            model = GoogLeNet()
        case 'LeNet':
            model = LeNet()
        case 'MyNet':
            model = MyNet()
        case 'ResNet':
            model = ResNet()
        case 'VGG':
            model = VGG()
        case _:
            raise NameError('Unknown model type')
    model.to(device)

    ### dataloader
    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'),
                                  batch_size=args.batch_size,
                                  split='train')
    val_loader = get_dataloader(os.path.join(dataset_dir, 'val'),
                                batch_size=args.batch_size,
                                split='val')

    criterion = nn.CrossEntropyLoss()
    if args.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          logfile_dir=logfile_dir,
          model_save_dir=model_save_dir,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device)


if __name__ == '__main__':
    args = get_args()
    main(args)
