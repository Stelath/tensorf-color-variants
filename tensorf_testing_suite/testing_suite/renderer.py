import torch

def render_scene(tensorf, vector, size=512, device='cuda'):
    testing_rays = []
    for i in range(size):
        for j in range(size):
            testing_rays.append([1.5-j/size*2, 1.5-i/size*2, -1.7500, -0.5, -0.5,  2.0000])

    testing_rays = torch.tensor(testing_rays, dtype=torch.float32).to(device)
    testing_rays = testing_rays.view(-1, int(4096/size)*size, 6)

    output = []
    with torch.no_grad():
        tensorf.eval()
        for ray in testing_rays:
            output.append(tensorf(ray)[0].cpu().numpy())
    
    rgb_map = np.array(output)
    rgb_map = rgb_map.reshape(size,size,3)
    
    return rgb_map
