import warnings

import matplotlib.pyplot as plt
import cv2
import mmcv
import random
from PIL import Image


import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from openselfsup.models import build_model
from openselfsup.utils import build_from_cfg
from openselfsup.datasets.registry import PIPELINES

from torchvision.transforms import Compose


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_model(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = Image.open(results['img'])
        img = img.convert('RGB')
        return img


def inference_model(model, img):
    """Inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    img = Image.open(img)
    img = img.convert('RGB')
    data = test_pipeline(img)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(data, mode='test')
    return result

