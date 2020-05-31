import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--load_model_path', help='Path to model(.pt file)', type=str)

    parser = parser.parse_args(args)

    # dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
    #                           transform=transforms.Compose([Normalizer(), Resizer()]))

    dataset_val = CocoDataset(parser.coco_path, set_name='val',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = torch.load(parser.load_model_path)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
