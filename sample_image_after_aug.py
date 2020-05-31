
import os
import cv2
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.dataloader import CocoDataset, AugmenterWithImgaug, Normalizer, UnNormalizer, Resizer, AspectRatioBasedSampler, collater


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser = parser.parse_args(args)

    os.makedirs('after_augmentation_image_sample', exist_ok=True)

    set_name = 'test'

    dataset_sample = CocoDataset(parser.coco_path, set_name=set_name,
                                    # transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
                                    transform=transforms.Compose([Normalizer(), AugmenterWithImgaug(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_sample, batch_size=1, drop_last=False)
    dataloader_sample = DataLoader(dataset_sample, num_workers=1, collate_fn=collater, batch_sampler=sampler)

    unnormalize = UnNormalizer()

    for idx, data in enumerate(dataloader_sample):


        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

        img[img < 0] = 0
        img[img > 255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for annot in data['annot']:
            annot = annot[0].data.numpy()
            x1 = int(annot[0])
            y1 = int(annot[1])
            x2 = int(annot[2])
            y2 = int(annot[3])

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imwrite('D:\\StreetView\\RenHong\\Pytorch_RetinaNet\\after_augmentation_image_sample\\' + 'sample_from_({})_'.format(set_name) + str(idx) + '.jpg', img)

    print("finish")
if __name__ == '__main__':
    main()