import argparse
import os
import json
from pathlib import Path
import numpy as np

import torch

from backend.skinvision.models.networks import LMDAUNet
from backend.skinvision.checkpoints import load_model_weights
from backend.skinvision.models.export_ops import replace_deform_convolutions


def main():
    parser = argparse.ArgumentParser(description="Export LMDAU-Net to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default="outputs/model.onnx", help="Output ONNX path")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version, >=16 for portable GridSample")
    parser.add_argument("--verify", action="store_true", help="Compare ONNX Runtime output with the original native model")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()
    if args.opset<16:
        parser.error('Portable deformable convolution requires opset >=16')
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261119)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = LMDAUNet()
    load_model_weights(model, args.weights, device)
    model.to(device)
    model.eval()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        expected=model(dummy,mode='test')[3].cpu().numpy() if args.verify else None
    replaced=replace_deform_convolutions(model)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            outputs = self.net(x, mode="test")
            return outputs[3]

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["mask_logits"],
        dynamic_axes={"input": {0: "batch"}, "mask_logits": {0: "batch"}},
    )
    print(f"Exported ONNX model to: {args.output}")
    if args.verify:
        import onnx
        import onnxruntime as ort
        onnx.checker.check_model(args.output)
        options=ort.SessionOptions()
        options.intra_op_num_threads=args.threads
        session=ort.InferenceSession(args.output,sess_options=options,providers=['CPUExecutionProvider'])
        actual=session.run(None,{'input':dummy.cpu().numpy()})[0]
        error=float(np.abs(actual-expected).max())
        np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=2e-4)
        metadata={'input_shape':list(dummy.shape),'output':'fourth-head mask_logits (legacy semantics)',
                  'opset':args.opset,'max_abs_error':error,'replaced_deform_convolutions':replaced,
                  'torch_version':torch.__version__,'onnxruntime_version':ort.__version__,
                  'validated_batch':1,'spatial_dimensions':'fixed at export','dynamic_batch':'declared, not validated beyond batch1'}
        Path(args.output+'.validation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps(metadata))


if __name__ == "__main__":
    main()
