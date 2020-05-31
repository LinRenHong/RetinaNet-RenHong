
import os
import time
import datetime
import argparse
import collections
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from retinanet import model
from retinanet import coco_eval
from retinanet import csv_eval
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer, AugmenterWithImgaug
from retinanet.utils import printProgressBar, chop_microseconds

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument("--load_model_path", type=str, default=None, help="Path to model (.pt) file.")
    parser.add_argument('--dataset_type', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--backbone', help='Backbone choice: [ResNet, ResNeXt]', type=str, default='ResNet')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-5, help="adam: learning rate")

    parser = parser.parse_args(args)

    results_dir = "results"
    save_images_dir = os.path.join(results_dir, "images")
    save_models_dir = os.path.join(results_dir, "saved_models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_models_dir, exist_ok=True)

    # Get today datetime
    today = datetime.date.today()
    today = "%d%02d%02d" % (today.year, today.month, today.day)

    # Get current timme
    now = time.strftime("%H%M%S")

    # Backbone name
    backbone_name = parser.backbone + str(parser.depth)

    # DataSet name
    dataset_path = ''

    # Create the data loaders
    if parser.dataset_type == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        # dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
        #                             transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        # dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
        #                           transform=transforms.Compose([Normalizer(), Resizer()]))

        dataset_train = CocoDataset(parser.coco_path, set_name='train',
                                    # transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
                                    transform=transforms.Compose([Normalizer(), AugmenterWithImgaug(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

        dataset_path = parser.coco_path

    elif parser.dataset_type == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

        dataset_path = parser.csv_train

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Retrain the model
    if parser.load_model_path is not None:
        # Load pretrained models
        print("\nLoading model from: [%s]" % parser.load_model_path)
        retinanet = torch.load(parser.load_model_path)
        print("\nStart retrain...")
    # Create the model
    else:
        print("\nStart train...")
        if parser.backbone == 'ResNet':
            if parser.depth == 18:
                retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
            elif parser.depth == 34:
                retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
            elif parser.depth == 50:
                retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
            elif parser.depth == 101:
                retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
            elif parser.depth == 152:
                retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
            else:
                raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        elif parser.backbone == 'ResNeXt':
            if parser.depth == 50:
                retinanet = model.resnext50_32x4d(num_classes=dataset_train.num_classes(), pretrained=True)
            elif parser.depth == 101:
                retinanet = model.resnext101_32x8d(num_classes=dataset_train.num_classes(), pretrained=True)
                pass
            else:
                raise ValueError("Unsupported model depth, must be one of 50, 101")

        else:
            raise ValueError("Choice a backbone, [ResNet, ResNeXt]")

    # Get dataset name
    dataset_name = os.path.split(dataset_path)[-1]

    # Checkpoint name
    save_ckpt_name = r"%s_%s-%s-RetinaNet-backbone(%s)-ep(%d)-bs(%d)-lr(%s)" \
                     % (today, now, dataset_name, backbone_name, parser.epochs, parser.batch_size, parser.lr)

    os.makedirs(os.path.join(save_images_dir, "%s" % save_ckpt_name), exist_ok=True)
    os.makedirs(os.path.join(save_models_dir, "%s" % save_ckpt_name), exist_ok=True)

    tb_log_path = os.path.join("tf_log", save_ckpt_name)
    tb_writer = SummaryWriter(os.path.join(results_dir, tb_log_path))

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    val_loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    epoch_prev_time = time.time()
    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        total_classification_loss = 0.0
        total_regression_loss = 0.0
        total_running_loss = 0.0

        total_val_classification_loss = 0.0
        total_val_regression_loss = 0.0
        total_val_running_loss = 0.0

        batch_prev_time = time.time()
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

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

                # sum the loss for tensorboard at this batch
                total_regression_loss += regression_loss
                total_classification_loss += classification_loss
                total_running_loss += loss.item()


                # log = 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))

                # Determine approximate time left
                data_done = iter_num
                data_left = len(dataloader_train) - data_done
                batch_time_left = datetime.timedelta(seconds=data_left * (time.time() - batch_prev_time))
                batch_time_left = chop_microseconds(batch_time_left)

                batches_done = epoch_num * len(dataloader_train) + iter_num
                batches_left = parser.epochs * len(dataloader_train) - batches_done
                total_time_left = datetime.timedelta(seconds=batches_left * (time.time() - epoch_prev_time))
                total_time_left = chop_microseconds(total_time_left)

                batch_prev_time = time.time()
                epoch_prev_time = time.time()

                # Print training step log
                prefix_log = '[Epoch: {}/{}] | [Batch: {}/{}]'.format(epoch_num + 1, parser.epochs, iter_num + 1, len(dataloader_train))
                suffix_log = '[Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}] ETA: {} / {}'.format(
                    float(classification_loss), float(regression_loss), np.mean(loss_hist), batch_time_left, total_time_left)

                printProgressBar(iteration=iter_num + 1, total=len(dataloader_train), prefix=prefix_log, suffix=suffix_log)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        # Validation
        with torch.no_grad():
            val_batch_prev_time = time.time()
            for iter_num, data in enumerate(dataloader_val):
                try:

                    val_classification_loss, val_regression_loss = retinanet(
                        [data['img'].cuda().float(), data['annot']])

                    val_classification_loss = val_classification_loss.mean()
                    val_regression_loss = val_regression_loss.mean()

                    val_loss = val_classification_loss + val_regression_loss

                    if bool(val_loss == 0):
                        continue

                    val_loss_hist.append(float(val_loss))

                    # sum the loss for tensorboard at this batch
                    total_val_regression_loss += val_regression_loss
                    total_val_classification_loss += val_classification_loss
                    total_val_running_loss += val_loss.item()

                    # Determine approximate time left
                    data_done = iter_num
                    data_left = len(dataloader_val) - data_done
                    val_batch_time_left = datetime.timedelta(
                        seconds=data_left * (time.time() - val_batch_prev_time))
                    val_batch_time_left = chop_microseconds(val_batch_time_left)

                    batches_done = epoch_num * len(dataloader_val) + (epoch_num + 1) * len(
                        dataloader_train) + iter_num
                    batches_left = parser.epochs * (len(dataloader_train) + len(dataloader_val)) - batches_done
                    total_time_left = datetime.timedelta(seconds=batches_left * (time.time() - epoch_prev_time))
                    total_time_left = chop_microseconds(total_time_left)

                    val_batch_prev_time = time.time()
                    epoch_prev_time = time.time()

                    # Print training step log
                    prefix_log = 'Validation: [Epoch: {}/{}] | [Batch: {}/{}]'.format(epoch_num + 1,
                                                                                      parser.epochs,
                                                                                      iter_num + 1,
                                                                                      len(dataloader_val))
                    suffix_log = '[Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}] ETA: {} / {}'.format(
                        float(val_classification_loss), float(val_regression_loss), np.mean(val_loss_hist),
                        val_batch_time_left,
                        total_time_left)

                    printProgressBar(iteration=iter_num + 1, total=len(dataloader_val), prefix=prefix_log,
                                     suffix=suffix_log)

                    del val_classification_loss
                    del val_regression_loss
                except Exception as e:
                    print(e)
                    continue

        # Evaluate AP
        if parser.dataset_type == 'coco':

            print('Evaluating dataset')

            # coco_eval.evaluate_coco(dataset_val, retinanet)
            coco_eval.evaluate_coco_and_save_image(dataset_val, retinanet, os.path.join(save_images_dir, save_ckpt_name), epoch_num + 1)

        elif parser.dataset_type == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        # calculate loss average
        average_classification_loss = total_classification_loss / len(dataloader_train)
        average_regression_loss = total_regression_loss / len(dataloader_train)
        average_running_loss = total_running_loss / len(dataloader_train)

        # TensorBoard
        tb_writer.add_scalar(tag='Classification Loss', scalar_value=average_classification_loss, global_step=epoch_num + 1)
        tb_writer.add_scalar(tag='Regression Loss', scalar_value=average_regression_loss, global_step=epoch_num + 1)
        tb_writer.add_scalar(tag='Total Loss', scalar_value=average_running_loss, global_step=epoch_num + 1)

        # Save model
        print("\nSave model to [%s] at %d epoch\n" % (save_ckpt_name, epoch_num + 1))
        checkpoint_path = os.path.join(save_models_dir, "%s/RetinaNet_backbone(%s)_%d.pt" % (save_ckpt_name, backbone_name, epoch_num + 1))
        torch.save(retinanet.module, checkpoint_path)
        # torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset_type, epoch_num + 1))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
