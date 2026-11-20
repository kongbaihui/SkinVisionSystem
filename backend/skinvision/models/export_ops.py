"""Export-only bilinear sampling equivalent for the model's deformable convolutions."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d


class PortableDeformConv2d(nn.Module):
    def __init__(self, native):
        super().__init__()
        if native.groups != 1:
            raise ValueError('Portable export currently supports weight groups=1')
        self.weight, self.bias = native.weight, native.bias
        self.stride, self.padding, self.dilation = native.stride, native.padding, native.dilation
        self.kernel_size = native.kernel_size

    def forward(self, image, offset, mask=None):
        batch, channels, height, width = image.shape
        out_h, out_w = offset.shape[-2:]
        kh, kw = self.kernel_size
        if offset.shape[1] != 2*kh*kw:
            raise ValueError('Portable export requires one offset group')
        yy,xx = torch.meshgrid(torch.arange(out_h,device=image.device,dtype=image.dtype),
                               torch.arange(out_w,device=image.device,dtype=image.dtype),indexing='ij')
        samples=[]
        for row in range(kh):
            for column in range(kw):
                index=row*kw+column
                y=yy*self.stride[0]-self.padding[0]+row*self.dilation[0]+offset[:,2*index]
                x=xx*self.stride[1]-self.padding[1]+column*self.dilation[1]+offset[:,2*index+1]
                grid=torch.stack(((x+.5)/width*2-1,(y+.5)/height*2-1),dim=-1)
                sampled=F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
                if mask is not None:
                    sampled=sampled*mask[:,index:index+1]
                samples.append(sampled)
        sampled=torch.stack(samples,dim=2).reshape(batch,channels*kh*kw,out_h*out_w)
        output=torch.matmul(self.weight.reshape(self.weight.shape[0],-1).unsqueeze(0),sampled)
        output=output.reshape(batch,self.weight.shape[0],out_h,out_w)
        if self.bias is not None:
            output=output+self.bias[None,:,None,None]
        return output


def replace_deform_convolutions(module):
    count=0
    for name,child in list(module.named_children()):
        if isinstance(child,DeformConv2d):
            setattr(module,name,PortableDeformConv2d(child))
            count+=1
        else:
            count+=replace_deform_convolutions(child)
    return count
