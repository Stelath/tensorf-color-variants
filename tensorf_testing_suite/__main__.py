import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from tensorf_testing_suite.tensorf_models.tensoRF import TensorVMSplit
from tensorf_testing_suite.dataLoader.your_own_data import YourOwnDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    ckpt = torch.load('trained_models/tensorf_small_baseline.th', map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)
    
    return tensorf

def setup():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

def get_ray_data(tensorf, ray):
    ray = ray.to(device).view(-1, 6).to(device)
    print(ray)

    with torch.no_grad():
        tensorf.eval()
        output = tensorf.forward(ray)

    rgb_map, depth, rgb, sigma, xyz_sampled = output

    rgb = rgb[0].cpu().numpy()
    sigma = sigma[0].cpu().numpy()
    xyz_sampled = xyz_sampled[0].cpu().numpy()
    
    return rgb, sigma, xyz_sampled

def get_color_variance_

def main():
    setup()
    
    model = load_model()
    
    dataset = YourOwnDataset('./datasets/small_baseline/', split='test', is_stack=True)
    rays = dataset[0]['rays']
    
    for ray in rays:
        rgb, sigma, xyz_sampled = get_color_variance(model, ray)
    

if __name__ == "__main__":
    main()
