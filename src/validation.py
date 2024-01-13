import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
from src.loss_function import energy_loss
from sklearn.metrics import confusion_matrix

def get_accuracy(preds, ground_truth):
    ground_truth = ground_truth.squeeze(dim=1)
    preds = preds.argmax(dim=1)
    
    return (preds.flatten()==ground_truth.flatten()).float().mean()

def validation(model, val_set):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    val_dataloader = DataLoader(val_set, batch_size=5, shuffle = True)
    loss_function = None
    y_true = []
    y_pred = []
    weight = torch.Tensor([1.33524479, 142.03737758, 354.33279529, 121.55201728,
       170.52369266, 173.57602029,  59.18592147,  73.39980364,
        39.04301533,  91.24823152, 124.53864632,  80.32893704,
        62.08797479, 112.79122179,  92.20176115,  21.86262213,
       161.68561906, 118.22250115,  72.47050034,  65.89660941,
       116.10541954])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    weight = weight.to(device)
    accuracies = []
    losses = []

    with tqdm(val_dataloader) as tepoch:

        for imgs, smnts in tepoch:
            
            with torch.no_grad():
                out = model(imgs.to(device))
            
            smnts *= 255
            smnts = torch.where(smnts==255, 0, smnts)

            accuracy = get_accuracy(out, smnts.to(device))
            loss = energy_loss(out, smnts.to(device), weight = weight)  
            tepoch.set_postfix(accuracy=accuracy.item(), loss=loss.item())  
            losses.append(loss.item())
            accuracies.append(accuracy.item())

            # out_flat = smnts.squeeze(dim=1).flatten().tolist()
            y_pred.extend(out.argmax(dim = 1).flatten().cpu().tolist())
            y_true.extend(smnts.squeeze(dim=1).flatten().tolist())

    print (f"Accuracy: {sum(accuracies)/len(accuracies)}")
    print (f"Validation Loss: {sum(losses)/len(losses)}")
    # cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    # print (cm)
    # print (f"F1 Score: f1_score(y_true, y_pred, average = 'weighted')")

    return sum(accuracies)/len(accuracies), sum(losses)/len(losses)

    # cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    # cm.to_csv("val_results/results.csv")
    # return f1_score(y_true, y_pred, average = 'weighted')

if __name__ == "__main__":
    
    from src.dataset import get_load_data

    _, val_set = get_load_data(root = "../data", dataset = "VOCSegmentation")
    trained_model_path = "model_weights/model_weights.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))
    validation(model, val_set)
            