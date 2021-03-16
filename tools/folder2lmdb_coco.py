# coding: utf-8

import argparse
import os
import os.path as osp
import lmdb
import pyarrow as pa
import json

import torch.utils.data as data
from torch.utils.data import DataLoader


class COCODataset(data.Dataset):
    """
    pass.

    """

    def __init__(self, infos, dpath):
        file_names = list()
        for info in infos:
            file_names.append(info["file_name"])

        self.file_names = file_names
        self.dpath = dpath

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = raw_reader(osp.join(self.dpath, file_name))
        return img


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


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


def folder2lmdb(infos, dpath, name="train2017", workers=32, write_frequency=1000):
    data_path = osp.expanduser(osp.join(dpath, name))
    dataset = COCODataset(infos, data_path)
    data_loader = DataLoader(dataset, num_workers=workers)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, image in enumerate(data_loader):
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(image[0]))
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
    parser = argparse.ArgumentParser(description='COCO Folder to LMDB.')
    parser.add_argument('--data-root', type=str,
                        default='/home/datasets/coco/',
                        help='the name of data root.')
    parser.add_argument('--ann-root', type=str,
                        default='',
                        help='the name of ann root.')
    parser.add_argument('--data-type', type=str,
                        default='train2017',
                        help='the name of data type.')
    parser.add_argument('--num-worker', type=int, default=64)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataRoot = args.data_root
    if args.ann_root:
        annRoot = args.ann_root
    else:
        annRoot = osp.join(dataRoot, "annotations")
    dataType = args.data_type
    numWorker = args.num_worker

    annFile = osp.join(annRoot, "instances_{}.json".format(dataType))
    # load annotations.
    with open(annFile, 'r') as f:
        annJson = json.load(f)
    # we only use the img id.
    infos = annJson["images"]
    del annJson

    # process.
    folder2lmdb(infos, dpath=dataRoot, name=dataType, workers=numWorker)
