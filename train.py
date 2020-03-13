import argparse
import collections
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import VOCDetection, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from retinanet import coco_eval
from retinanet import csv_eval
assert torch.__version__.split('.')[0] == '1'
from tqdm.autonotebook import tqdm
import shutil
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='RegiGraph Pytorch Implementation Training Script. - Ahmed Nassar (ETHZ, IRISA).')
    parser.add_argument("--batch_size", type=int, default=4, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--dataset_root', default='../datasets', help='Dataset root directory path [../datasets/VOC, ../datasets/mapillary]')
    parser.add_argument('--dataset', default='Pasadena', choices=['Pasadena', 'Pasadena_Aerial', 'mapillary'],
                        type=str, help='Pasadena, Pasadena_Aerial or mapillary')
    parser.add_argument("--overfit", type=int, default="0")
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_path", type=str, default="tensorboard/")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--cluster", type=int, default=0)
    
    opt = parser.parse_args(args)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if(opt.dataset == 'Pasadena' or opt.dataset == 'mapillary' or opt.dataset == 'Pasadena_Aerial'):
        train_dataset = VOCDetection(root=opt.dataset_root, overfit= opt.overfit, image_sets="trainval", transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), dataset_name=opt.dataset)
        valid_dataset = VOCDetection(root=opt.dataset_root, overfit= opt.overfit, image_sets="val", transform=transforms.Compose([Normalizer(), Resizer()]), dataset_name=opt.dataset)

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(train_dataset, batch_size=2, drop_last=False)

    training_params = {"batch_size": opt.batch_size,
                   "shuffle": True,
                   "drop_last": True,
                   "collate_fn": collater,
                   "num_workers": 4}

    training_generator = DataLoader(train_dataset, **training_params)

    if valid_dataset is not None:
        test_params = {"batch_size": opt.batch_size,
               "shuffle": False,
               "drop_last": False,
               "collate_fn": collater,
               "num_workers": 4}
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        test_generator = DataLoader(valid_dataset, **test_params)

    # Create the model
    if opt.depth == 18:
        retinanet = model.resnet18(num_classes=train_dataset.num_classes(), pretrained=True)
    elif opt.depth == 34:
        retinanet = model.resnet34(num_classes=train_dataset.num_classes(), pretrained=True)
    elif opt.depth == 50:
        retinanet = model.resnet50(num_classes=train_dataset.num_classes(), pretrained=True)
    elif opt.depth == 101:
        retinanet = model.resnet101(num_classes=train_dataset.num_classes(), pretrained=True)
    elif opt.depth == 152:
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

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)


    retinanet.training = True
    writer = SummaryWriter(opt.log_path+"regigraph_bs_"+str(opt.batch_size)+"_dataset_"+opt.dataset+"_backbone_"+str(opt.depth))
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    best_loss = 1e5
    best_epoch = 0

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(train_dataset)))

    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        progress_bar = tqdm(training_generator)

        for iter, data in enumerate(progress_bar):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'], data['batch_map']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot'], data['batch_map']])
                    
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
                total_loss = np.mean(epoch_loss)
                
                if opt.cluster == 0:
                    progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, classification_loss, regression_loss, float(loss),
                        total_loss))
                    writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                    writer.add_scalar('Train/Regression_loss', regression_loss, epoch * num_iter_per_epoch + iter)
                    writer.add_scalar('Train/Classfication_loss (focal loss)', classification_loss, epoch * num_iter_per_epoch + iter)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.test_interval == 0:
            retinanet.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_generator):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'], data['batch_map']])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss_classification_ls.append(float(classification_loss))
                    loss_regression_ls.append(float(regression_loss))

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, opt.num_epochs, cls_loss, reg_loss, np.mean(loss)))
            writer.add_scalar('Test/Total_loss', loss, epoch)
            writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + opt.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                mAP = csv_eval.evaluate(valid_dataset, retinanet)
                print(mAP)
                torch.save(retinanet.module, os.path.join(opt.saved_path, "regigraph_bs_"+str(opt.batch_size)+"_dataset_"+opt.dataset+"_epoch_"+str(epoch+1)+"_backbone_"+str(opt.depth)+".pth"))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break
    writer.close()

if __name__ == '__main__':
    main()
