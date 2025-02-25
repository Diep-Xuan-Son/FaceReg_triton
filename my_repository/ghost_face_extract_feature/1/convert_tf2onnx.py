# import tensorflow as tf
# import tf2onnx
# import onnx

# model = tf.saved_model.load("model.savedmodel")
# input_signature = [tf.TensorSpec([112, 112], tf.float32, name='x')]
# # Use from_function for tf functions
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=16)
# onnx.save(onnx_model, "model.onnx")





#-----------test----------
import onnxruntime
import numpy as np
import time

batch_size = 10
x = np.random.rand(batch_size, 112, 112, 3).astype(np.float32)

st_time = time.time()
# for _ in range(10):
ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)
print(f"----Duration: {time.time() - st_time}")