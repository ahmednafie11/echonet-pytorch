import echonet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

def train():
    # Load dataset
    dataset = echonet.datasets.Echo(root="data", split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, 
num_workers=4)

    # Load model
    model = r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # For EF regression
    model = model.cuda() if torch.cuda.is_available() else model

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for batch in dataloader:
            frames = batch["frame"].cuda() if torch.cuda.is_available() 
else batch["frame"]
            ef = batch["ef"].cuda() if torch.cuda.is_available() else 
batch["ef"]
            optimizer.zero_grad()
            output = model(frames).squeeze()
            loss = criterion(output, ef)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
