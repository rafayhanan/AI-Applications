import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


img_path = "Images/walllpaper.jpg"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)