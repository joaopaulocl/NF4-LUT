import torch
import torch.nn as nn

from bitsandbytes.nn import Linear4bit, Linear4bitFakeQuantAct, LinearNF4Compute

torch.manual_seed(24)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

input_dtype = torch.float16 if device.type == "cuda" else torch.float32

N = 64
fp16_model = nn.Sequential(
    nn.Linear(N, N),
    nn.Linear(N, N)
).to(device=device, dtype=input_dtype)

quantized_model = nn.Sequential(
    Linear4bit(N, N,  quant_type="nf4"),
    Linear4bit(N, N,  quant_type="nf4")
)

full_quantized_model = nn.Sequential(
    Linear4bitFakeQuantAct(N, N,  quant_type="nf4"),
    Linear4bitFakeQuantAct(N, N,  quant_type="nf4")
)

nf4_model = nn.Sequential(
    LinearNF4Compute(N, N, ),
    LinearNF4Compute(N, N, )
)

print("nf4_model[0] type:", type(nf4_model[0]))
print("has load_state_dict:", hasattr(nf4_model[0], 'load_state_dict'))
print("load_state_dict method:", nf4_model[0].load_state_dict)

quantized_model.load_state_dict(fp16_model.state_dict())
quantized_model = quantized_model.to(device)  # Quantization happens here

full_quantized_model.load_state_dict(fp16_model.state_dict())
full_quantized_model = full_quantized_model.to(device)  # Quantization happens here

print("fp16 state_dict keys:", list(fp16_model.state_dict().keys()))
print("nf4 state_dict keys:", list(nf4_model.state_dict().keys()))


nf4_model[0].load_state_dict({'weight': fp16_model.state_dict()['0.weight'], 'bias': fp16_model.state_dict()['0.bias']})
nf4_model[1].load_state_dict({'weight': fp16_model.state_dict()['1.weight'], 'bias': fp16_model.state_dict()['1.bias']})
#nf4_model[0].load_state_dict({'weight': fp16_model.state_dict()['0.weight']})
#nf4_model[1].load_state_dict({'weight': fp16_model.state_dict()['1.weight']})
nf4_model = nf4_model.to(device)  # Quantization happens here

sample_input = torch.randn(1, N, device=device, dtype=input_dtype)

with torch.inference_mode():
    fp16_out = fp16_model(sample_input)
    quant_out = quantized_model(sample_input)
    full_quant_out = full_quantized_model(sample_input)
    nf4_out = nf4_model(sample_input)

print("fp16 output:", fp16_out.flatten()[:10])
print("quantized output:", quant_out.flatten()[:10])
print("full quantized output:", full_quant_out.flatten()[:10])
print("nf4 output:", nf4_out.flatten()[:10])

print("fp16 output norm:", fp16_out.norm().item())
print("quantized output norm:", quant_out.norm().item(), "Max diff:", (fp16_out - quant_out).abs().max().item())
print("full quantized output norm:", full_quant_out.norm().item(), "Max diff:", (fp16_out - full_quant_out).abs().max().item())
print("nf4 output norm:", nf4_out.norm().item(), "Max diff:", (fp16_out - nf4_out).abs().max().item())