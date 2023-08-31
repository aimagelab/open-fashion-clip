import open_clip
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('weights/openfashionclip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

#Load the image
img = Image.open('examples/maxi_dress.jpg')
img = preprocess(img).to(device)

prompt = "a photo of a"
text_inputs = ["blue cowl neck maxi-dress", "red t-shirt", "white shirt"]
text_inputs = [prompt + " " + t for t in text_inputs]

tokenized_prompt = tokenizer(text_inputs).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(img.unsqueeze(0)) #Input tensor should have shape (b,c,h,w)
    text_features = clip_model.encode_text(tokenized_prompt)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Labels probs:", text_probs)