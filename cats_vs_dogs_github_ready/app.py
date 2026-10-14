import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os

# Ensure paths are correct for local imports
sys.path.append(os.getcwd())

try:
    from catsdogs.model import get_model
except ImportError:
    try:
        from model import get_model
    except ImportError:
        import torchvision.models as models
        def get_model(arch='resnet18', num_classes=2):
            model = getattr(models, arch)(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['cat', 'dog']
model = get_model(arch='resnet18', num_classes=2)

# Correctly load from checkpoint dict
checkpoint = torch.load('models/best_model.pt', map_location=device)
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def energy_score(logits, T=1.0):
    # Lower energy = more confident inside distribution. High energy = Out of Distribution.
    return -T * torch.logsumexp(logits / T, dim=1)

def predict_image(image, threshold, energy_threshold):
    if image is None:
        return "No image uploaded."

    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype('uint8'), 'RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        # Calculate free energy score
        energy = energy_score(logits).item()
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

    # Check for Out-of-Distribution
    if energy > energy_threshold:
         return {"Out of Distribution (OOD)": 1.0, "cat": 0.0, "dog": 0.0}

    cat_prob = float(probabilities[0])
    dog_prob = float(probabilities[1])

    max_prob = max(cat_prob, dog_prob)
    best_class = 'cat' if cat_prob > dog_prob else 'dog'

    if max_prob < threshold:
        return f"Low Confidence ({max_prob:.1%}). Might be a {best_class}, but is below the {threshold:.1%} threshold."

    return {"cat": cat_prob, "dog": dog_prob}

# Setup Gradio Interface with Sliders for Confidence & Energy Threshold
interface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(label="Upload Cat or Dog Image"),
        gr.Slider(minimum=0.5, maximum=1.0, value=0.7, step=0.05, label="Confidence Threshold"),
        gr.Slider(minimum=-15.0, maximum=15.0, value=-3.0, step=0.5, label="OOD Energy Threshold (Higher = More strict OOD filter)")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Cats vs Dogs Classifier with Energy OOD Detection",
    description="Includes standard Confidence classification, plus Energy-based Out-of-Distribution detection for unknown images."
)

if __name__ == '__main__':
    interface.launch(share=True)
