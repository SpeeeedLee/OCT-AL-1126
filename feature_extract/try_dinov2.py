from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests


# https://github.com/facebookresearch/dinov2/issues/153
# https://huggingface.co/facebook/dinov2-large


cache_dir = '/storage/ssd3/ArthurLee/HuggingFace' 
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large', cache_dir = cache_dir)
model = AutoModel.from_pretrained('facebook/dinov2-large', cache_dir = cache_dir)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape) # torch.Size([1, 257, 1024])

# get the CLS embedding for each image in the batch
cls_embeddings = last_hidden_states[:, 0, :]      # shape: [1, 1024]
print(cls_embeddings.shape)