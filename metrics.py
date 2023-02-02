import numpy as np
import torch
from data_visualization import plot_pair

def calculate_iou(y, t):
    y = (y > 0.5).to(torch.float32)
    t = t.round()
    
    intersection = (y * t).sum((2, 3))
    union = ((y + t) >= 1).to(torch.float32).sum((2, 3))

    return (intersection / union).mean()