import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.eeg_audio_fusion import EEGAudioFusionModel
from utils.data_loader import EEGAudioDataset
from utils.logger import Logger

def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = EEGAudioDataset(config["data"]["processed"])
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    model = EEGAudioFusionModel(
        config["model"]["eeg_dim"],
        config["model"]["audio_dim"],
        config["model"]["hidden_dim"],
        config["model"]["num_heads"],
        config["model"]["num_classes"]
    )
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    logger = Logger(config["train"]["log_dir"])

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0
        for eeg, audio, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(eeg, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.log(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

train()
