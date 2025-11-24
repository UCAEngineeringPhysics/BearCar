import copy
import torch
from cnn_architectures.bear_net import BearNet
from torchao.quantization import Int4WeightOnlyConfig, quantize_

model = BearNet().eval().to(torch.bfloat16).to("cuda")

# Optional: compile model for faster inference and generation
model = torch.compile(model, mode="max-autotune", fullgraph=True)
model_bf16 = copy.deepcopy(model)

quantize_(model, Int4WeightOnlyConfig(group_size=32, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq"))
