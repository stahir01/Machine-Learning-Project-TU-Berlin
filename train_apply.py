import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_loading import load_data
from model import NewUNet

from train import train_model, test_model
from entropy_loss import DiceBCELoss

#NUM_EPOCHS = 50
#LR = 0.07
#MOMENTUM = 99
#BATCH_SIZE = 2

#SUFFIX = f'EP_{NUM_EPOCHS}_LR_{LR}_MOM_{MOMENTUM}_BS_{BATCH_SIZE}'

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        bce_weight = 0.5
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final

def train_apply(method = 'train_model',dataset = 'isbi_em_seg', num_epochs=25, lr=0.01, momentum=0.99, batch_size=3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    train_loader, test_loader = load_data(dataset=dataset, batch_size=batch_size)
    model = NewUNet(1).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)   
    criterion = DiceBCELoss()

    eval(f'{method}(model, train_loader, test_loader, optimizer, criterion, device, num_epoch={num_epochs})')

    predictions, mask, confidence_score = test_model(model, test_loader, device)

    return predictions, mask, confidence_score

def predict(learning_rate=0.01, epochs=25, batch_size=3, momentum=0.99):
    predictions, mask, confidence_score = train_apply(lr=learning_rate, num_epochs=epochs, batch_size=batch_size, momentum=momentum)
    print("Confidence score: ", confidence_score)

    prediction_test = predictions#[:, 1:2, :, :]
    mask_test = mask


    #Final Result
    fig = plt.figure(figsize=(20,20))
    for i in range (len(prediction_test)):
      while i < 2:

        predict_results = prediction_test[i].cpu().numpy()
        mask_results = mask_test[i].cpu().numpy()


        #predict_results = (predict_results * 255.0).astype("uint8")
        #mask_results = (mask_results * 255.0).astype("uint8")

        predict_results = predict_results[0]
        mask_results = mask_results[0]

        predict_results = predict_results.squeeze()
        mask_results = mask_results.squeeze()
        #print("Predict Results: ", predict_results)
        #print("Mask Results: ", mask_results)

        plt.subplot(5,2,2*i+1)
        plt.imshow(mask_results, cmap = 'gray')
        plt.axis("off")
        plt.subplot(5,2,2*i+2)
        plt.imshow(predict_results, cmap = 'gray') 
        plt.axis("off")
        i+=1
    plt.show()




if __name__ == '__main__':
    NUM_EPOCHS = 10
    LR = 0.07
    MOMENTUM = 0.99
    BATCH_SIZE = 2

    predict(learning_rate=LR, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, momentum=MOMENTUM)