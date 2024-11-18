import torch
from torch.utils.data import DataLoader
from models.eeg_audio_fusion import EEGAudioFusionModel
from utils.data_loader import EEGAudioDataset
from utils.metrics import calculate_accuracy

def evaluate():
    dataset = EEGAudioDataset("data/processed/")
    dataloader = DataLoader(dataset, batch_size=32)
    model = EEGAudioFusionModel(eeg_dim=128, audio_dim=40, hidden_dim=256, num_heads=4, num_classes=10)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for eeg, audio, labels in dataloader:
            outputs = model(eeg, audio)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total * 100:.2f}%")

evaluate()
