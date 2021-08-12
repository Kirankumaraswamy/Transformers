import torch

cuda_is_available = torch.cuda.is_available()
cuda_current_device = torch.cuda.current_device() if cuda_is_available else None
cuda_device = torch.cuda.device(0) if cuda_is_available else None
cuda_device_count = torch.cuda.device_count() if cuda_is_available else None
cuda_get_device_name = torch.cuda.get_device_name(0) if cuda_is_available else None
GLIBCXX_USE_CXX11_ABI = torch._C._GLIBCXX_USE_CXX11_ABI

if __name__ == "__main__":
    print("cuda_is_available: ", cuda_is_available)  # cuda_is_available:  True
    print("cuda_current_device: ", cuda_current_device)  # cuda_current_device:  0
    print("cuda_device: ", cuda_device)  # cuda_device:  <torch.cuda.device object at 0x7fd95d9ad040>
    print("cuda_device_count: ", cuda_device_count)  # cuda_device_count:  1
    print("cuda_get_device_name: ", cuda_get_device_name)  # GeForce GTX 980 Ti
    print("GLIBCXX_USE_CXX11_ABI: ", GLIBCXX_USE_CXX11_ABI)