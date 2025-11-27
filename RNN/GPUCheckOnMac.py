import tensorflow as tf
from tensorflow.python.client import device_lib

print(f"TensorFlow Version: {tf.__version__}")

# check available device list
available_devices = device_lib.list_local_devices()
print("\nAvailable Devices:")
for x in available_devices:
    # Check if there GPU available
    if 'GPU' in x.name or 'METAL' in x.name:
        print(f"✅ FOUND ACCELERATOR: {x.name} ({x.device_type})")
        
# confirm it working.
if tf.config.list_physical_devices('GPU'):
    print("\n🎉 Success! TensorFlow is configured to use the GPU/Metal backend.")
else:
    print("\n⚠️ Warning: No GPU detected. Check installation steps.")