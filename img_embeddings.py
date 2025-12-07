import os
import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

query = False

# Paths
if query:
    output_path = r"E:/Projects/Honours/query_im_end_embeddings.npy"
    img_dir = r"E:/Projects/Honours/market1501/market-1501-v15.09.15/query"
else:
    output_path = r"E:/Projects/Honours/gallery_im_end_embeddings.npy"
    img_dir = r"E:/Projects/Honours/market1501/market-1501-v15.09.15/bounding_box_test"

print(f"Image directory: {img_dir}")

# Load model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
print("Model loaded")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print("Model and processor loaded.")

# Get <|im_end|> token ID
im_end_token_str = "<|im_end|>"
im_end_token_id = processor.tokenizer.convert_tokens_to_ids(im_end_token_str)

print(f"<|im_end|> token ID: {im_end_token_id}")

# List query images
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")][6500:8000]

print(f"Number of images to process: {len(img_files)}")
print("first 10 image files: ", img_files[:10])

embeddings = {}

# Try to load existing embeddings if file exists
if os.path.exists(output_path):
    embeddings = np.load(output_path, allow_pickle=True).item()

for img_file in img_files:
    if img_file in embeddings:
        print(
            f"Skipping {img_file}, already processed. Shape: {embeddings[img_file].shape}"
        )
        continue
    img_path = os.path.join(img_dir, img_file)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": "Summarize the person image, focusing on age, gender, clothing, and biometric features",
                },
            ],
        }
    ]
    raw_img = Image.open(img_path)
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    emb = outputs.hidden_states
    input_ids = inputs["input_ids"][0]
    im_end_positions = (input_ids == im_end_token_id).nonzero(as_tuple=True)[0]
    if len(im_end_positions) > 0:
        im_end_pos = im_end_positions.item()
        im_end_embedding = emb[-1][0, im_end_pos, :].to(torch.float32).cpu().numpy()
        print(f"Embedding shape for {img_file}: {im_end_embedding.shape}")
        embeddings[img_file] = im_end_embedding
        np.save(output_path, embeddings)
        print(
            f"Extracted and saved embedding for {img_file}, shape: {im_end_embedding.shape}. Total: {len(embeddings)}"
        )
    else:
        print(f"<|im_end|> token not found in {img_file}")
