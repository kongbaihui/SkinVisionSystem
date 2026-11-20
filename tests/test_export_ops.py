import torch
from torchvision.ops import DeformConv2d


def test_portable_deform_conv_matches_native_values_and_input_gradient():
    from backend.skinvision.models.export_ops import PortableDeformConv2d
    torch.manual_seed(20)
    native=DeformConv2d(2,3,3,padding=1).double()
    portable=PortableDeformConv2d(native)
    image=torch.randn(2,2,5,7,dtype=torch.double,requires_grad=True)
    offset=torch.randn(2,18,5,7,dtype=torch.double,requires_grad=True)*.4
    expected=native(image,offset)
    actual=portable(image,offset)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
    g1=torch.autograd.grad(expected.square().sum(),(image,offset),retain_graph=True)
    g2=torch.autograd.grad(actual.square().sum(),(image,offset))
    for first,second in zip(g1,g2):
        torch.testing.assert_close(first,second,rtol=1e-7,atol=1e-7)
