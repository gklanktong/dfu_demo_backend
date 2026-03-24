import torch
import torch.nn as nn
from torchvision.models import resnet50

MODEL_PATH = "models/resnet_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = resnet50(weights=None)

    # MUST MATCH TRAINING EXACTLY
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model