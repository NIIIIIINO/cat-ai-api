import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)
model.classifier = torch.nn.Identity()
model = model.to(device).eval()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def get_embedding(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.cpu().numpy().flatten()
