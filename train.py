import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import numpy as np

from data_loading import load_data
from model import NewUNet

from tqdm import tqdm

from matplotlib.pyplot import imshow
from data_visualization import plot_pair

from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt

from metrics import calculate_iou

def get_test_metrics(model, test_loader, criterion, device):
   losses, ious, confidences = [], [], []
   with torch.no_grad():
      for i, data in enumerate(test_loader):

         image, mask = data[0].to(device), data[1].to(device)
         output = model(image)
         output = output[:,1].unsqueeze(dim=1)
         
         loss = criterion(output, mask)
         losses.append(loss.item())

         iou = calculate_iou(output, mask)
         ious.append(iou.item())

   return torch.tensor(losses).mean().cpu().numpy().astype(float), torch.tensor(ious).mean().cpu().numpy().astype(float)

def custom_loss(output, mask):
   return (output - mask).abs().mean(dim=(1,2,3)).mean()
   # return ((output - mask)**2).mean(dim=(1,2,3)).norm()

def test_model(model, test_loader, device):
   model.eval()

   predictions, confidences = [], []
   with torch.no_grad():
      for i, data in enumerate(tqdm(test_loader)):
         image, mask = data[0].to(device), data[1].to(device)
         output = model(image)

         probabilities = torch.softmax(output, dim=1)
         
         #Calculate confidence score
         confidence, index = torch.max(probabilities, dim=1)
         confidence_mean = torch.mean(confidence).item()


         confidences.append(confidence_mean)
         predictions.append(output)
         


   predictions = torch.cat(predictions, dim=0)
   confidences = np.array(confidences)

   return predictions, mask, confidences.mean()

def train_one_epoch(model, train_loader, optimizer, criterion, device):
   losses, ious = [], []
   for i, (image, mask) in enumerate(tqdm(train_loader, leave=False)):
      image, mask = image.to(device), mask.to(device)

      optimizer.zero_grad()

      output = model(image)
      output = output[:,1].unsqueeze(dim=1)

      loss = criterion(output, mask)
      iou = calculate_iou(output, mask)
      losses.append(loss.item())
      ious.append(iou.item())
      loss.backward()
      optimizer.step()

   avg_loss = torch.tensor(losses).mean().cpu().numpy().astype(float)
   avg_iou = torch.tensor(ious).mean().cpu().numpy().astype(float)
   return avg_loss, avg_iou
         
def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epoch = 100):
   model.train()

   avg_ious_train, avg_ious_test = [], []
   avg_losses_train, avg_losses_test = [], []

   for epoch in range(num_epoch): 
      avg_loss_train, avg_iou_train = train_one_epoch(model, train_loader, optimizer, criterion, device)
      avg_loss_test, avg_iou_test= get_test_metrics(model, test_loader, criterion, device)

      avg_ious_train.append(avg_iou_train)
      avg_ious_test.append(avg_iou_test)
      avg_losses_train.append(avg_loss_train)
      avg_losses_test.append(avg_loss_test)
      print("Epoch {0}: train_loss {1} \t train_score {2} \t test_loss {3} \t test_score{4}".format(epoch, avg_loss_train, avg_iou_train, avg_loss_test, avg_iou_test))
      
   plt.figure(figsize=(25,5))
   plt.plot(avg_losses_train, label='Training Loss')
   plt.plot(avg_ious_train, label='Training IoU')
   plt.plot(avg_losses_test, label='Testing Loss')
   plt.plot(avg_ious_test, label='Testing IoU')
   plt.xlabel('epochs')
   plt.ylabel('Avg Loss & Score')
   plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
   plt.title('Train Loss & Train Score')
   plt.legend(loc='best')
   plt.grid(visible=True, which='both')
   plt.show()