import onnxruntime as ort
import torch

def get_onnx_provider():
    """Get the appropriate ONNX Runtime provider based on the available hardware."""
    available_providers = ort.get_available_providers()
    
    if torch.backends.mps.is_available() and 'CoreMLExecutionProvider' in available_providers:
        return ['CoreMLExecutionProvider']
    elif 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider']
    else:
        return ['CPUExecutionProvider'] 