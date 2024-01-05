import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

def one_hot_encode(smnt: torch.Tensor, num_classes) -> torch.Tensor:
    
    B, C, H, W = smnt.shape    
    # change 255 to 0 since we don't care about the border
    smnt = torch.where(smnt == 255, 0, smnt)
        
    smnt_one_hot = one_hot(smnt.to(torch.int64), num_classes = num_classes + 1)
    
    # permute and reshape to get the correct shape
    smnt_one_hot = smnt_one_hot.permute(0, 1, 4, 2, 3)
    smnt_one_hot = smnt_one_hot.reshape(B, num_classes + 1, H, W)
    
    # remove the background class
    # smnt_one_hot = smnt_one_hot[:, :, 1:, :, :]

    return smnt_one_hot

def energy_loss(pred = torch.Tensor, ground_truth= torch.Tensor) -> torch.Tensor:
    """Energy loss
    Supposed to follow the following steps
    1. Softmax function along the channels
    2. Compute the cross entropy loss when compared to the ground truth
    3. Weighted sum of the loss

    But somehow torch.nn.CrossEntropyLoss() does all of this for you. So, we just need to call it

    Args:
        pred (_type_, optional): pred. 0 is always background. Hence, if you will have n_classes + 1 channels. Defaults to torch.Tensor.
        ground_truth (_type_, optional): Ground Truth. If the mask has 255 it is ignored. Needs to be int64 type. Defaults to torch.Tensor.

    Returns:
        torch.Tensor: _description_
    """
    if ground_truth.dtype != torch.int64:
        ground_truth = ground_truth.type(torch.int64)

    loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    return loss(pred, ground_truth.squeeze(1))

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