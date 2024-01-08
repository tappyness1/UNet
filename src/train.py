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


def train(train_set, cfg, in_channels = 3, num_classes = 10):

    loss_function = None # using energy_loss instead
    
    network = UNet(img_size = 572, num_classes = num_classes + 1)

    network.train()

    if cfg['show_model_summary']:
        summary(network, (in_channels,572,572))

    optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle = True)
    
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, smnts in tepoch:
                # print (imgs.shape)
                optimizer.zero_grad() 
                out = network(imgs.to(device))
                loss = energy_loss(out, smnts.to(device))
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
           'train': {"epochs": 2, 'lr': 0.001, 'weight_decay': 5e-5}}
    
    train_set, test_set = get_load_data(root = "../data", dataset = "VOCSegmentation")
    train(train_set = train_set, cfg = cfg, in_channels = 3, num_classes = 20)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    