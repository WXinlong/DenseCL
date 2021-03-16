# coding: utf-8

import argparse
import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pyarrow as pa
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def is_valid_file(filename):
    x = "/".join(filename.split("/")[-2:])
    if x in valid_list.keys() and x.lower().endswith(".jpeg"):
        if valid_list[x]:
            flag = True
            valid_list[x] = False
        else:
            flag = False
    else:
        flag = False
    return flag


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="train", workers=32, write_frequency=5000):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader, is_valid_file=is_valid_file)
    data_loader = DataLoader(dataset, num_workers=workers, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet Folder to LMDB.')
    parser.add_argument('--data-root', type=str,
                        default='/home/datasets/imagenet/',
                        help='the name of data root.')
    parser.add_argument('--list-path', type=str,
                        default='/home/datasets/imagenet/meta',
                        help='the name of list path.')
    parser.add_argument('--data-type', type=str,
                        default='val',
                        help='the name of data type.')
    parser.add_argument('--num-worker', type=int, default=64)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataRoot = args.data_root
    listPath = args.list_path
    dataType = args.data_type
    numWorker = args.num_worker

    # generate valid list.
    listFile = os.path.join(listPath, dataType + ".txt")
    with open(listFile, "r") as f:
        lines = f.readlines()
    global valid_list
    valid_list = dict()
    for line in lines:
        valid_list[line.split()[0]] = True

    # process.
    folder2lmdb(dpath=dataRoot, name=dataType, workers=numWorker)
