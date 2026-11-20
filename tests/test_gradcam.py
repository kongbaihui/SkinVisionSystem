import numpy as np
import torch
from backend.skinvision.pipeline import SkinVisionPipeline


class TinySegmentation(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.out_head1 = torch.nn.Conv2d(3, 1, 1)
        self.fail = fail

    def forward(self, x, mode='test'):
        y = self.out_head1(x)
        if self.fail:
            raise RuntimeError('simulated forward failure')
        return [y] * 4


def test_gradcam_does_not_accumulate_parameter_gradients():
    pipeline = SkinVisionPipeline(use_cpu=True)
    pipeline.model = TinySegmentation().eval()
    image = np.ones((3, 8, 8), dtype=np.float32)
    result = pipeline._generate_gradcam(image, np.zeros((8, 8, 3), dtype=np.uint8))
    assert result is not None and result.shape == (8, 8, 3)
    assert all(p.grad is None for p in pipeline.model.parameters())
    assert not pipeline.model.out_head1._forward_hooks
    assert not pipeline.model.out_head1._backward_hooks


def test_gradcam_removes_hook_on_failure():
    pipeline = SkinVisionPipeline(use_cpu=True)
    pipeline.model = TinySegmentation(fail=True).eval()
    assert pipeline._generate_gradcam(np.ones((3, 8, 8), dtype=np.float32), np.zeros((8, 8, 3), dtype=np.uint8)) is None
    assert not pipeline.model.out_head1._forward_hooks
    assert not pipeline.model.out_head1._backward_hooks
