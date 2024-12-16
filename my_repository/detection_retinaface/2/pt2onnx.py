import numpy as np
import torch
import torch.nn as nn

torch_model = torch.load("model.pt", map_location="cpu")
batch_size = 1
x = torch.randn(batch_size, 3, 640, 640, requires_grad=True)
# print(x.dtype)
torch_out = torch_model(x)
# print(torch_out[0].shape)
# exit()
# Export the model
torch_model = torch_model.eval()
# print(torch_model)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model1.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['inputs'],   # the model's input names
                  output_names = ['output_0','output_1','output_2'], # the model's output names
                  dynamic_axes={'inputs' : {0 : 'batch_size'},    # variable length axes
                                'output_0' : {0 : 'batch_size'},
                                'output_1' : {0 : 'batch_size'},
                                'output_2' : {0 : 'batch_size'},})

exit()

#----------------onnx32 -> onnx16------------
import onnx
from onnxconverter_common import float16
model = onnx.load("model.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "model_fp16.onnx")
exit()
#////////////////////////////////////////////

# #--------------test onnx------------
# import onnxruntime
# import numpy as np

# torch_out = torch_model(torch_input)

# ort_session = onnxruntime.InferenceSession("detectFace_model1.onnx", providers=["CPUExecutionProvider"])
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# def to_numpy2(tensor):
#     return tensor[0].detach().cpu().numpy() if tensor[0].requires_grad else tensor[0].cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch_input)}
# ort_outs = ort_session.run(None, ort_inputs)
# print(ort_outs[0].shape)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy2(torch_out), ort_outs[0], rtol=1e-03, atol=1e-5)
# exit()

#-------------test trt------------
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(TRT_LOGGER)
# builder = trt.Builder(TRT_LOGGER)
# exit()

# Deserialize the engine from file
engine_file_path = "model.trt"
with open(engine_file_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
# print(context)
print(engine.__dir__())
for binding in engine:
    print('bingding:', binding, engine.get_tensor_shape(binding))
    print(trt.volume(engine.get_tensor_shape(binding)), trt.nptype(engine.get_tensor_dtype(binding)))

