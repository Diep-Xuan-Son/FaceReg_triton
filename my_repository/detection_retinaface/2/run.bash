trtexec --onnx=model.onnx --saveEngine=model.trt 
# --shapes=inputs:64x3x640x640

# the command below still fail
#trtexec --onnx=model.onnx --saveEngine=model.plan --minShapes=inputs:64x3x112x112 --maxShapes=inputs:64x3x640x640 --fp16
