import torch
import torch.nn as nn
import torch.optim as optim
from model import CifarClassifier
from utils import get_loaders, log_results
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resolutions = [16, 24, 32, 48, 64]
epochs = 10

for res in resolutions:
    print(f"\n--- Training for resolution {res}x{res} ---")

    trainloader, testloader = get_loaders(res, subset_ratio=1.0)
    model = CifarClassifier(input_size=res).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)
        print(f"[{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")

    log_results(res, acc, train_losses[-1], training_time)

    os.makedirs("results/plots", exist_ok=True)
    plt.plot(train_losses)
    plt.title(f"Loss curve (res={res})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"results/plots/loss_curve_{res}.png")
    plt.close()
