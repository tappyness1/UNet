import torch
from torch.nn import BCEWithLogitsLoss

def energy_loss(pred = torch.Tensor, ground_truth= torch.Tensor) -> torch.Tensor:

    # perform softmax function here?

    # perform cross entropy loss here?

    # calculate the weights for each class here?

    loss = None

    return loss

if __name__ == "__main__":
    from src.model import UNet
    from src.dataset import get_load_data

    unet_model = UNet(num_classes=20)
    train, test = get_load_data(root = "../data", dataset = "VOCSegmentation", download = False)  
    img, smnt = train[0] 
    img = img.reshape(1, 3, 572, 572)

    smnt = smnt.resize((388, 388))

    # change smnt to torch.Tensor here
    smnt = torch.Tensor(np.asarray(smnt)).reshape(1, 1, 388, 388)

    pred = unet_model(img)

    loss = energy_loss(pred, smnt)