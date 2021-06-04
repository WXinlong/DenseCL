from math import cos, pi
from mmcv.runner import Hook

from .registry import HOOKS


@HOOKS.register_module
class DenseCLWarmupHook(Hook):
    '''Hook in DenseCLWarmupHook
    This hook including loss_lambda warmup in DenseCL following:
    '''

    def __init__(self, start_iters=1000, **kwargs):
        self.start_iters = start_iters

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCLWarmupHook."
        self.loss_lambda = runner.model.module.loss_lambda

    def before_train_iter(self, runner):
        assert hasattr(runner.model.module, 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCLWarmupHook."
        cur_iter = runner.iter
        if cur_iter >= self.start_iters:
            runner.model.module.loss_lambda = self.loss_lambda
        else:
            runner.model.module.loss_lambda = 0.
