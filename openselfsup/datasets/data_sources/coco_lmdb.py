# coding: utf-8
import os.path as osp
from PIL import Image
import six
import lmdb
import pyarrow as pa

import torch.utils.data as data

from ..registry import DATASOURCES


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


@DATASOURCES.register_module
class CocoLMDB(data.Dataset):
    def __init__(self, root, list_file):
        db_path = osp.join(root, list_file)
        self.db_path = db_path
        self.env = None
        if 'train2017' in self.db_path:
            self.length = 118287
        elif 'val2017' in self.db_path:
            self.length = 5000
        else:
            raise NotImplementedError

    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()

        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = loads_pyarrow(byteflow)

        # load img.
        imgbuf = unpacked
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        return img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def get_length(self):
        return self.length

    def get_sample(self, idx):
        return self.__getitem__(idx)
