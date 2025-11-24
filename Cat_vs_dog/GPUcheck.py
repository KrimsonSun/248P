import tensorflow as tf
import time

print("--- TensorFlow 内部 GPU 检查 ---")

# 尝试列出所有物理 GPU 设备
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # 设置内存增长，避免分配过多 GPU 内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"Find {len(gpus)}  Gpu!")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # 运行一个简单的矩阵乘法来测试 GPU 加速
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        
        # 如果矩阵乘法在 GPU 上执行，则说明加速成功
        print("\nGPU Test (tf.matmul):")
        print(c.numpy())
        print("GPU is Varified, It works on GPU")

    except RuntimeError as e:
        print(f"Error,something else cause error: {e}")
else:
    print("TensorFlow GPU not found。")

print("-----------------------------------")