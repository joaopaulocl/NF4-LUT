import torch
import torch.nn as nn

from bitsandbytes.nn import Linear4bit, Linear4bitFakeQuantAct

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

input_dtype = torch.float16 if device.type == "cuda" else torch.float32

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
).to(device=device, dtype=input_dtype)

quantized_model = nn.Sequential(
    Linear4bit(64, 64, quant_type="nf4"),
    Linear4bit(64, 64, quant_type="nf4")
)

full_quantized_model = nn.Sequential(
    Linear4bitFakeQuantAct(64, 64, quant_type="nf4"),
    Linear4bitFakeQuantAct(64, 64, quant_type="nf4")
)

quantized_model.load_state_dict(fp16_model.state_dict())
quantized_model = quantized_model.to(device)  # Quantization happens here

full_quantized_model.load_state_dict(fp16_model.state_dict())
full_quantized_model = full_quantized_model.to(device)  # Quantization happens here

sample_input = torch.randn(1, 64, device=device, dtype=input_dtype)

with torch.inference_mode():
    fp16_out = fp16_model(sample_input)
    quant_out = quantized_model(sample_input)
    full_quant_out = full_quantized_model(sample_input)

print("fp16 output norm:", fp16_out.norm().item())
print("quantized output norm:", quant_out.norm().item())
print("full quantized output norm:", full_quant_out.norm().item())