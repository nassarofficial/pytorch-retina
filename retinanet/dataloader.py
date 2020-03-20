from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import os.path as osp
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
from random import shuffle


from pycocotools.coco import COCO
import cv2
import skimage.io
import skimage.transform
import skimage.color
import skimage
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import pickle
from PIL import Image


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, dataset_name,class_to_ind=None, keep_difficult=False):
        self.dataset_name = dataset_name
        if self.dataset_name == 'Pasadena' or "Pasadena_Aerial":
            self.class_to_ind = {'tree':0}
        elif self.dataset_name == 'mapillary':
            self.class_to_ind = {'sign':0}

        self.keep_difficult = keep_difficult

    def normalizer(self, x, mode, dataset):
        if dataset == "Pasadena":
            minn,maxx=0,0
            if mode == 'yaw':
                minn,maxx = 0.0, 360
            elif mode == 'pitch':
                minn,maxx = 0.03, 9.46
            elif mode == 'pano_lat':
                minn,maxx = 34.12397, 34.177644
            elif mode == 'pano_lng':
                minn,maxx = -118.185097, -118.073197
            elif mode == 'obj_lat':
                minn,maxx = 34.12406212011486, 34.17751982080085
            elif mode == 'obj_lng':
                minn,maxx = -118.18505646290065, -118.07330317391417
            elif mode == 'width':
                minn,maxx = 0,13312
            elif mode == 'height':
                minn,maxx = 0,6656
            normalized = (x-minn)/(maxx-minn)
        else:
            # Mapillary Normalization
            minn,maxx=0,0
            if mode == 'yaw':
                minn,maxx = 0.0, 360
            elif mode == 'pano_lat':
                minn,maxx = 34.12397, 34.177644
            elif mode == 'pano_lng':
                minn,maxx = -0.40344755026555484, 0.11468149304376465
            elif mode == 'obj_lat':
                minn,maxx = 51.402448659857235, 51.610936663987154
            elif mode == 'obj_lng':
                minn,maxx = -0.40361261546491295, 0.1148465169449277
            elif mode == 'width':
                minn,maxx = 0,13312
            elif mode == 'height':
                minn,maxx = 0,6656
            normalized = (x-minn)/(maxx-minn)
        return normalized

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []

        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            ID = int(obj.find('ID').text)

            tree_location = obj.find('location').text

            if tree_location == "None":
                latitude = -1
                longitude = -1
            else:
                tree_location = tree_location.split(",")
                latitude = self.normalizer(float(tree_location[1]),"obj_lat",self.dataset_name)
                longitude = self.normalizer(float(tree_location[0]),"obj_lng",self.dataset_name)


            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            bndbox.append(ID)
            bndbox.append(latitude)
            bndbox.append(longitude)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

            img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, dataset_name, root, image_sets, overfit,
                 transform=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform=VOCAnnotationTransform(dataset_name)
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'Images', '%s.jpg')
        self.ids = []
        self.rootpath = osp.join(self.root, self.name)

        if self.name == "Pasadena" or "Pasadena_Aerial":
            self.VOC_CLASSES = ["tree"]
        elif self.name == "mapillary":
            self.VOC_CLASSES = ["sign"]

        dataset_dir = 'Main'
        if overfit == 1:
            dataset_dir = 'Main_Overfit'

        self.ids_full = pickle.load( open( osp.join(self.rootpath, 'ImageSets', dataset_dir, self.image_set + '.p'), "rb" ) )
        # self.ids_full = []
        # sub_ids = []
        # for line in open(osp.join(self.rootpath, 'ImageSets', 'Main', self.image_set + '.txt')):
        #     sub_ids.append((self.rootpath, line.strip()))
        #     if len(sub_ids) == 4:
        #         self.ids_full.append(sub_ids)
        #         sub_ids = []
    def normalizer(self, x, mode, dataset):
        if dataset == "Pasadena":
            minn,maxx=0,0
            if mode == 'yaw':
                minn,maxx = 0.0, 360
            elif mode == 'pitch':
                minn,maxx = 0.03, 9.46
            elif mode == 'pano_lat':
                minn,maxx = 34.12397, 34.177644
            elif mode == 'pano_lng':
                minn,maxx = -118.185097, -118.073197
            elif mode == 'obj_lat':
                minn,maxx = 34.12406212011486, 34.17751982080085
            elif mode == 'obj_lng':
                minn,maxx = -118.18505646290065, -118.07330317391417
            elif mode == 'width':
                minn,maxx = 0,13312
            elif mode == 'height':
                minn,maxx = 0,6656
            normalized = (x-minn)/(maxx-minn)
        else:
            # Mapillary Normalization
            minn,maxx=0,0
            if mode == 'yaw':
                minn,maxx = 0.0, 360
            elif mode == 'pano_lat':
                minn,maxx = 34.12397, 34.177644
            elif mode == 'pano_lng':
                minn,maxx = -0.40344755026555484, 0.11468149304376465
            elif mode == 'obj_lat':
                minn,maxx = 51.402448659857235, 51.610936663987154
            elif mode == 'obj_lng':
                minn,maxx = -0.40361261546491295, 0.1148465169449277
            elif mode == 'width':
                minn,maxx = 0,13312
            elif mode == 'height':
                minn,maxx = 0,6656
            normalized = (x-minn)/(maxx-minn)
        return normalized

    def __getitem__(self, index):
        img_ids = self.ids_full[index]
        items = []
        for i in range(len(img_ids)):
            if self.name == "Pasadena":
                target = ET.parse(self._annopath % (self.rootpath, img_ids[i]+"_z2")).getroot()
                img = cv2.imread(self._imgpath % (self.rootpath, img_ids[i]+"_z2"))
                geo = pickle.load(open(osp.join(self.rootpath,"GeoFeats",img_ids[i]+".p"), "rb" ))

            else:
                target = ET.parse(self._annopath % (self.rootpath, img_ids[i])).getroot()
                img = cv2.imread(self._imgpath % (self.rootpath, img_ids[i]))
                geo = pickle.load(open(osp.join(self.rootpath,"GeoFeats",img_ids[i]+".p"), "rb" ))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.

            pano_lat = self.normalizer(float(geo['lat']),"pano_lat",self.name)
            pano_lng = self.normalizer(float(geo['lng']),"pano_lng",self.name)
            yaw = self.normalizer(float(geo['yaw']),"yaw",self.name)
            geo_dat = np.array([pano_lat,pano_lng,yaw])

            height, width, channels = img.shape

            if self.target_transform is not None:
                target = self.target_transform(target, width, height)

            target = np.array(target)
            sample = {'img': img, 'annot': target, 'geo': geo_dat}
            if self.transform is not None:
                sample = self.transform(sample)
                items.append(sample)
            else:
                bbox = target[:, :4]
                labels = target[:, 4]
                if self.transform is not None:
                    annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
                    augmentation = self.transform(**annotation)
                    img = augmentation['image']
                    bbox = augmentation['bboxes']
                    labels = augmentation['category_id']
                    items.append({'image': img, 'bboxes': bbox, "geo":geo_dat, 'category_id': labels})
        return items

    def __len__(self):
        return len(self.ids_full)

    def num_classes(self):
        return len(self.VOC_CLASSES)

    def label_to_name(self, label):
        return self.VOC_CLASSES[label]

    def load_annotations(self, index):
        img_ids = self.ids_full[index]
        gts = []
        for i in range(len(img_ids)):
            anno = ET.parse(self._annopath % (self.rootpath, img_ids[i])).getroot()
            gt = self.target_transform(anno, 1, 1)
            gt = np.array(gt)
            gts.append(gt)
        return gts



def collater(data):
    # implement for multiple batch sizes
    imgs = []
    annots = []
    scales = []
    geos = [] 
    batch_map = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            imgs.append(data[i][j]['img'])
            annots.append(data[i][j]['annot'])
            scales.append(data[i][j]['scale'])
            geos.append(data[i][j]['geo'])
        batch_map[i] = len(data[i])
        # imgs.append(data[i][0]['img'])
        # annots.append(data[i][0]['annot'])
        # scales.append(data[i][0]['scale'])

    # print("123 :", len(imgs))
    # print("123 :", imgs[0])
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 8)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 8)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'geo': geos, 'batch_map': batch_map}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots, geos = sample['img'], sample['annot'], sample['geo']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'geo': torch.from_numpy(geos)}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots, geos = sample['img'], sample['annot'], sample['geo']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'geo': geos}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots, geos = sample['img'], sample['annot'], sample['geo']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, 'geo': geos}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

# class RandomSamplerMod(Sampler):
#     r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
#     Drops the extra items, not fitting into exact batches
#     Arguments:
#         data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
#         batch_size (int): a batch size that you would like to use later with Dataloader class
#         shuffle (bool): whether to shuffle the data or not
#     """

#     def __init__(self, data_source, batch_size, shuffle=True):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = list(range(len(self.data_source.ids_full)))
#         self.n = len(self.indices)

#     def __iter__(self):
#         return iter(torch.randperm(self.n).tolist())


#     def __len__(self):
#         return len(self.data_source)
