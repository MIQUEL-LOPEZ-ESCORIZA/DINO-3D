import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should return '12.0' or similar
print(torch.cuda.get_device_name(0))  # Should return 'NVIDIA H100' or similar