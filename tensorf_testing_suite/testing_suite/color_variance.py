import torch
import numpy as np

from tensorf_models.tensoRF import TensorVMSplit

def variance_map(tensorf, xyz, device='cuda'):
    xyz = torch.tensor(xyz).to(device).view(1,3)

    sampled_ray = ray[0].cpu().numpy()
    testing_views = []
    size = 90
    for i in range(size**2):
        testing_views.append([(((i%size)-45)/360)+sampled_ray[3], ((int(i/size)-45)/360)+sampled_ray[4], sampled_ray[5]])

    testing_views = torch.tensor(testing_views, dtype=torch.float32).to(device)

    variance_map = []
    with torch.no_grad():
        tensorf.eval()
        for view in testing_views:
            testing_view = torch.tensor(view).to(device).view(1,3)
            variance_map.append(tensorf.get_variance(xyz, testing_view)[0].cpu().numpy())
        variance_map = np.array(variance_map)
        variance_map = variance_map.reshape(size,size,3)
    
    return variance_map