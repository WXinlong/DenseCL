import os
from PIL import Image
import glob

from ..registry import DATASOURCES
from .utils import McLoader


@DATASOURCES.register_module
class COCO(object):

    def __init__(self, root, list_file=None, memcached=False, mclient_path=None):
        self.fns = glob.glob(root + '/*g')
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        return img
