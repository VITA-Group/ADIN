from __future__ import print_function, division

from shutil import copy2

from datasetUtils.dataset import *
from datasetUtils.datasetStat import normalization
from pridUtils.random_erasing import RandomErasing

prid_data_root_dir = '/home/yeyuan/prid_data'
dataset_info = {
    "MSMT17": {
        "train": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "train"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_train.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "train_info.txt")},

        "val": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "train"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_val.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "val_info.txt")},

        "query": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "test"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_query.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "query_info.txt")},

        "gallery": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "test"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_gallery.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "gallery_info.txt")},
    },

    "DukeMTMC": {
        "train": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "train_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "train_Distractors_All_info.txt")},

        "val": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "val_info.txt")},

        "query": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "query"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "query_info.txt")},

        "gallery": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "bounding_box_test"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "gallery_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "gallery_Distractors_info.txt")},
    },

    "Market1501": {
        "train": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "train_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "train_Distractors_All_info.txt")},

        "val": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "val_info.txt")},

        "query": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "query"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "query_info.txt")},

        "gallery": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "bounding_box_test"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "gallery_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "gallery_Distractors_info.txt")},
    },

}


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


def getTransforms(use_erasing=False, use_colorjitter=False):
    data_transform = {'train': [transforms.Resize((432, 144), interpolation=3),
                                transforms.RandomCrop((384, 128))],
                      'val': [transforms.Resize(size=(384, 128), interpolation=3)]}

    data_transform['train'] = data_transform['train'] + [transforms.RandomHorizontalFlip()]

    for k in ['train', 'val']:
        data_transform[k] = data_transform[k] + [
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if use_erasing:
        data_transform['train'] = data_transform['train'] + [
            RandomErasing(probability=use_erasing[0.3, 0.3, 0.3], mean=[0.0, 0.0, 0.0])]

    # Randomly change the brightness, contrast and saturation of an image.
    if use_colorjitter:
        colorjitter = [0.3, 0.3, 0.3, 0]
        data_transform['train'] = [transforms.ColorJitter(
            brightness=colorjitter[0], contrast=colorjitter[1],
            saturation=colorjitter[2], hue=colorjitter[3])] + data_transform['train']

    return data_transform


def getDataloader(use_dataset, batch_size, log_dir):
    copy2("baseline/train/trainDataloaders.py", log_dir)

    data_transforms = getTransforms()

    train_datasets = {x: PRIDdataset(datasetInfo=dataset_info[use_dataset], subset=x,
                                     transform=transforms.Compose(data_transforms[x])) for x in ['train', 'val']}

    train_dataloaders = {x: torch.utils.data.DataLoader(train_datasets[x], batch_size=batch_size, shuffle=True,
                                                        num_workers=8) for x in train_datasets}

    return train_datasets, train_dataloaders


print("\nDataloader ... OK!\n")
