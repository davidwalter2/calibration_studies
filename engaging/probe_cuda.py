import ctypes, os, sys
for lib in ("libcuda.so.1", "libcudart.so.12", "libcudnn.so.9"):
    try:
        ctypes.CDLL(lib); print("OK   ", lib)
    except Exception as e:
        print("FAIL ", lib, e)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "<unset>"))
import tensorflow as tf
print("TF", tf.__version__, "built_with_cuda", tf.test.is_built_with_cuda())
from tensorflow.python.platform import build_info
print("build_info:", dict(build_info.build_info))
print("devices:", tf.config.list_physical_devices())
try:
    import tensorflow.python.framework.config as c
    print("gpu_device_name:", tf.test.gpu_device_name())
except Exception as e:
    print("gpu_device_name err", e)
