"""GPU configuration utilities for TensorFlow."""

from typing import Any

import tensorflow as tf


class GPUHandler:
    """
    A class to handle GPU configuration for TensorFlow.

    This class contains a static method to check the availability of GPUs
    and configure TensorFlow to allow memory growth on available GPUs.
    """

    @staticmethod
    def configure(config: Any) -> None:
        """
        Configures TensorFlow to use available GPUs.

        Parameters:
        - config: The configuration object.

        This method checks if any GPUs are available for TensorFlow. If available,
        it sets memory growth on each GPU to prevent TensorFlow from allocating all the
        GPU memory at once, allowing other processes to use the memory as needed.

        If the processor is 'M' (Mac M1/M2/M3), it configures TensorFlow accordingly.

        If memory growth cannot be set (e.g., because TensorFlow has already been
        initialized), it catches and prints the RuntimeError.
        """
        # Reduce TensorFlow verbosity
        import os
        import warnings

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all logs including warnings
        tf.get_logger().setLevel("ERROR")
        warnings.filterwarnings("ignore", category=UserWarning, module="keras")
        processor = getattr(config.hardware, "processor", "GPU")

        if processor == "M":
            # Configure TensorFlow for Apple M1/M2/M3 processors
            try:
                # Set up TensorFlow to use the 'metal' device (for Apple Silicon)
                tf.config.set_visible_devices([], "GPU")
                tf.config.set_visible_devices([], "XLA_GPU")
                tf.config.set_visible_devices([], "XLA_CPU")
                print("Configured TensorFlow for Apple Silicon processors.")
            except Exception as e:
                print("Could not set visible devices for M processor:", e)
        else:
            # Regular GPU configuration
            # Get the list of physical GPU devices recognized by TensorFlow
            gpus = tf.config.list_physical_devices("GPU")

            # Also check for CUDA availability
            cuda_available = tf.test.is_built_with_cuda()
            # Use modern method instead of deprecated is_gpu_available()
            gpu_available = len(gpus) > 0

            # Additional diagnostic info
            try:
                import subprocess

                nvidia_smi_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                nvidia_gpus = (
                    nvidia_smi_result.stdout.strip().split("\n")
                    if nvidia_smi_result.returncode == 0
                    else []
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                nvidia_gpus = []

            if gpus:
                try:
                    # Set memory growth FIRST before any GPU operations
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                    # Try to get GPU names using nvidia-ml-py if available
                    gpu_details = []
                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        for i in range(len(gpus)):
                            try:
                                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                                gpu_details.append(f"{name}")
                            except Exception:
                                gpu_details.append(f"GPU:{i}")
                    except ImportError:
                        # Fallback to TensorFlow device names
                        for i, gpu in enumerate(gpus):
                            try:
                                # Get GPU name from TensorFlow if available
                                gpu_name = (
                                    gpu.name.split("/")[-1]
                                    if hasattr(gpu, "name") and gpu.name
                                    else f"GPU:{i}"
                                )
                                gpu_details.append(gpu_name)
                            except Exception:
                                gpu_details.append(f"GPU:{i}")

                    if len(gpus) == 1:
                        print(f"🚀 GPU enabled: {gpu_details[0]}")
                    else:
                        print(
                            f"🚀 GPU enabled: {len(gpus)} devices ({', '.join(gpu_details)})"
                        )

                except RuntimeError as e:
                    # Catch and print exception if memory growth setting fails
                    print("Error setting GPU memory growth:", e)
            else:
                # More informative GPU status with diagnostics
                if cuda_available and not gpu_available:
                    print(
                        "💻 CPU mode: CUDA available but no GPUs accessible to TensorFlow"
                    )
                    if nvidia_gpus:
                        print(f"   🔍 nvidia-smi shows: {', '.join(nvidia_gpus)}")
                    print("   💡 Troubleshooting steps:")
                    print("      1. Check: nvidia-smi")
                    print(
                        '      2. Check TensorFlow build: python -c "import tensorflow as tf; print(tf.__version__, tf.test.is_built_with_cuda())"'
                    )
                    print("      3. For TF 2.12+: pip install tensorflow[and-cuda]")
                    print("      4. For older TF: pip install tensorflow-gpu")
                elif not cuda_available:
                    print("💻 CPU mode: TensorFlow not built with CUDA support")
                    print("   💡 Install GPU version: pip install tensorflow[and-cuda]")
                else:
                    print("💻 CPU mode: No GPUs detected")
