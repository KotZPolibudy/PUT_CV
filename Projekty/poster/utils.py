import torch
import torchvision
import torchvision.transforms as transforms
import time
import csv
import os
from torch.utils.data import Subset
import numpy as np
SEED = 69

def get_loaders(img_size, batch_size=64, subset_ratio=1.0):
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if subset_ratio < 1.0:
        np.random.seed(SEED)
        indices = np.random.choice(len(trainset), int(len(trainset) * subset_ratio), replace=False)
        trainset = Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper

def log_results(resolution, acc, loss, duration):
    os.makedirs('results', exist_ok=True)
    logfile = 'results/logs.csv'
    file_exists = os.path.isfile(logfile)
    with open(logfile, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Resolution', 'Accuracy', 'Loss', 'TrainingTime'])
        writer.writerow([resolution, acc, loss, duration])
