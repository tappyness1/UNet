import torch.nn as nn
from src.model import UNet
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf
from src.loss_function import energy_loss
from src.dataset import get_load_data
import numpy as np

def calc_weights(train_set):
    instances = []
    for i in range(len(train)):
        to_extend = train[i][1]
        to_extend *= 255
        instances.extend(to_extend.type(torch.int8).flatten().tolist())

    instances = np.array(instances)
    instances = np.where(instances == -1, 0, instances)

    counts = []
    for i in range(21):
        counts.append(np.count_nonzero(instances == i))
    counts = np.array(counts)
        
    weights = 1/ (counts / counts.sum())

    return torch.Tensor(weights)

def train(train_set, cfg, in_channels = 3, num_classes = 10):

    loss_function = None # using energy_loss instead

    # TODO: modify hardcoded weights
    weights = torch.Tensor([1.33524479, 142.03737758, 354.33279529, 121.55201728,
       170.52369266, 173.57602029,  59.18592147,  73.39980364,
        39.04301533,  91.24823152, 124.53864632,  80.32893704,
        62.08797479, 112.79122179,  92.20176115,  21.86262213,
       161.68561906, 118.22250115,  72.47050034,  65.89660941,
       116.10541954])
       
    weights = calc_weights(train_set)
    
    network = UNet(img_size = 572, num_classes = num_classes + 1)

    network.train()

    optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)
    weights = weights.to(device)

    if cfg['show_model_summary']:
        summary(network, (in_channels,572,572))

    train_dataloader = DataLoader(train_set, batch_size=5, shuffle = True)
    
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, smnts in tepoch:
                # print (imgs.shape)

                smnts = smnts * 255
                smnts = torch.where(smnts == 255, 0, smnts)

                optimizer.zero_grad() 
                out = network(imgs.to(device))
                loss = energy_loss(out, smnts.to(device), weight = weights)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
    print("training done")
    torch.save(network, cfg['save_model_path'])

    return network

if __name__ == "__main__":

    torch.manual_seed(42)

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': True, 
           'train': {"epochs": 20, 'lr': 0.005, 'weight_decay': 5e-3}}

    train_set, _ = get_load_data(root = "../data", dataset = "VOCSegmentation")
    train(train_set = train_set, cfg = cfg, in_channels = 3, num_classes = 20)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    