import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def get_model():
    global _model, _transform

    if _model is None:
        print("⏳ Loading MobileNet...")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        model.classifier = torch.nn.Identity()
        _model = model.to(device).eval()

        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✅ MobileNet ready")

    return _model, _transform


def get_embedding(img):
    model, transform = get_model()
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.cpu().numpy().flatten()
