import tensorflow as tf
import tf2onnx
import onnx

model = tf.saved_model.load("model.savedmodel")
input_signature = [tf.TensorSpec([112, 112], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=16)
onnx.save(onnx_model, "model.onnx")