import torch

def ray_data(tensorf, ray, device='cuda'):
    with torch.no_grad():
        tensorf.eval()
        output = tensorf.forward(ray)

    rgb_map, depth, rgb, sigma, xyz_sampled = output

    rgb = rgb[0].cpu().numpy()
    sigma = sigma[0].cpu().numpy()
    xyz_sampled = xyz_sampled[0].cpu().numpy()
    
    return rgb, sigma, xyz_sampled