import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import onnx
import numpy as np

# 1. DEFINE THE MODEL (The "Neural Net")
def create_model():
    model = tf.keras.Sequential([
        # 10 inputs (e.g., last 10 Close prices)
        layers.Dense(64, activation='relu', input_shape=(10,)), 
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear') # Predicts the next Close price
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 2. TRAIN & SAVE (Simplified for this example)
model = create_model()

# Save as .h5 (The file you mentioned)
model.save("MLP.REG.CLOSE.10000.h5")
print("Successfully created MLP.REG.CLOSE.10000.h5")

# 3. CONVERT TO ONNX (For MT5)
spec = (tf.TensorSpec((None, 10), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
onnx.save(model_proto, "MLP.REG.CLOSE.10000.onnx")

print("Successfully converted to MLP.REG.CLOSE.10000.onnx")
