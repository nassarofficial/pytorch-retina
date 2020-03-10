import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import VOCDetection, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument("--batch_size", type=int, default=4, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument('--dataset_root',
        default='/root/data/VOCdevkit/',
        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
    parser.add_argument('--dataset', default='Pasadena', choices=['Pasadena', 'mapillary'],
                        type=str, help='Pasadena or mapillary')
    parser.add_argument("--overfit", type=int, default="0")
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if(parser.dataset == 'Pasadena' or parser.dataset == 'mapillary'):
        train_dataset = VOCDetection(root=parser.dataset_root, overfit= parser.overfit, image_sets="trainval", transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), dataset_name=parser.dataset)
        valid_dataset = VOCDetection(root=parser.dataset_root, overfit= parser.overfit, image_sets="val", transform=transforms.Compose([Normalizer(), Resizer()]), dataset_name=parser.dataset)

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(train_dataset, batch_size=2, drop_last=False)

    training_params = {"batch_size": parser.batch_size * num_gpus,
                   "shuffle": True,
                   "drop_last": True,
                   "collate_fn": collater,
                   "num_workers": 12}

    training_generator = DataLoader(train_dataset, **training_params)

    if valid_dataset is not None:
        test_params = {"batch_size": parser.batch_size,
               "shuffle": False,
               "drop_last": False,
               "collate_fn": collater,
               "num_workers": 12}
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        test_generator = DataLoader(valid_dataset, **test_params)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=train_dataset.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=train_dataset.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=train_dataset.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=train_dataset.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=train_dataset.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(train_dataset)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(training_generator):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(valid_dataset, retinanet)

        else:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(valid_dataset, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
