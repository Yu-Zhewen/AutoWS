import argparse
import os
import pathlib
import re

from models.classification.imagenet import TorchvisionModelWrapper
from optimiser_interface.utils import opt_cli_launcher
from quantization.utils import QuantMode

parser = argparse.ArgumentParser()
parser.add_argument('--arch',  default='resnet18', help='model architecture', 
    choices=['resnet18', 'resnet50', 'mobilenetv2'])
parser.add_argument('--device', default='u250', help='FPGA device',
    choices=['u250', 'u50', 'zcu102', 'zc706', 'zedboard'])
parser.add_argument('--quantization', default='w4a4', help='quantization scheme')
parser.add_argument('--output_path', default=None, type=str, required=True,
                    help='output path')  
args = parser.parse_args()

pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

model_wrapper = TorchvisionModelWrapper(args.arch)
model_wrapper.load_model()
quant_config = {'weight_width': int(re.split("a|w|_", args.quantization)[1]),
                'data_width': int(re.split("a|w|_", args.quantization)[2]),
                'mode': QuantMode.CHANNEL_BFP}
model_wrapper.sideband_info['quantization'] = quant_config
model_wrapper.generate_onnx_files(os.path.join(args.output_path, f'{args.quantization}_bfp'))
opt_cli_launcher(args.arch, os.path.join(args.output_path, f'{args.quantization}_bfp', f'{args.arch}.onnx'),
                os.path.join(args.output_path, f'{args.quantization}_bfp_{args.device}'),
                batch_size=1, device=args.device)
