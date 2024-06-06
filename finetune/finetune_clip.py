import json
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

def load_captions(json_file, video_id):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Adjust this line if the structure is different
    captions = [item['caption'] for item in data['annotations'] if item['image_id'] == video_id]
    return captions

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

# Load captions
json_file = '/Users/achun/Downloads/atp-video-language/src/MSR_VTT.json'
video_id = 'video9997'
captions = load_captions(json_file, video_id)

# Load the frame as an image
image = Image.open('/home/adamchun/cs231n-project/finetune/atp-video-language/src/random_frame.jpg')
inputs = processor(text=[captions[0]], images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
# Model optimization setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# Forward pass
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score in the image dimension
logits_per_text = outputs.logits_per_text # this is the image-text similarity score in the text dimension
labels = torch.arange(inputs["input_ids"].shape[0], device=device)
loss = criterion(logits_per_image, labels) + criterion(logits_per_text, labels)
# Backward pass and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()


print(f"Training loss: {loss.item()}")
