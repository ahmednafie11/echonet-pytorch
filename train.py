import echonet
import torch
from torch.utils.data import DataLoader

def train():
    dataset = echonet.datasets.Echo(root="data", split="train")
    dataloader = DataLoader(dataset, batch_size=4)
    model = ...  # Load model
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch["frame"])
            loss = ...  # Compute loss
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
