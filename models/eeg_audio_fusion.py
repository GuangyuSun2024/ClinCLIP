import torch
import torch.nn as nn
from models.attention import MultiHeadAttention

class EEGAudioFusionModel(nn.Module):
    def __init__(self, eeg_dim, audio_dim, hidden_dim, num_heads, num_classes):
        super().__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, eeg_features, audio_features):
        eeg_out = self.eeg_encoder(eeg_features)
        audio_out = self.audio_encoder(audio_features)
        attention_out = self.attention(eeg_out, audio_out, audio_out)
        fusion_out = torch.cat((eeg_out, attention_out), dim=-1)
        fusion_out = self.dropout(self.fusion(fusion_out))
        return self.classifier(fusion_out)
