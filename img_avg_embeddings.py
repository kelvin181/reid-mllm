import os
import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

query = False  # Set to False for gallery

# Paths
if query:
    output_path = os.path.join(os.getcwd(), "query_im_avg_embeddings.npy")
    img_dir = os.path.join(os.getcwd(), "market1501", "market-1501-v15.09.15", "query")
else:
    output_path = os.path.join(os.getcwd(), "gallery_im_avg_embeddings.npy")
    img_dir = os.path.join(os.getcwd(), "market1501", "market-1501-v15.09.15", "bounding_box_test")
print(f"Image directory: {img_dir}")

# Load model and processor
device = torch.device("cpu")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct"
).to(device)
print("Model loaded")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print("Model and processor loaded.")

# Get <|im_end|> token ID
im_end_token_str = "<|im_end|>"
im_end_token_id = processor.tokenizer.convert_tokens_to_ids(im_end_token_str)

print(f"<|im_end|> token ID: {im_end_token_id}")

# List images
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

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
    # Move all tensor inputs to CPU
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    emb = outputs.hidden_states
    input_ids = inputs["input_ids"][0]
    # Find all <|im_end|> token positions
    im_end_positions = (input_ids == im_end_token_id).nonzero(as_tuple=True)[0]
    # Find all image token positions (token id == processor.tokenizer.image_token_id)
    # Use only <|im_end|> token for each image
    im_end_positions = (input_ids == im_end_token_id).nonzero(as_tuple=True)[0]
    num_im_end_tokens = len(im_end_positions)
    num_layers = len(emb)
    print(f"Number of layers: {num_layers}")
    if num_im_end_tokens == 1:
        layer_embeddings = []
        for layer in range(num_layers):
            # Take the <|im_end|> token embedding at this layer
            layer_emb = emb[layer][0, im_end_positions[0], :].to(torch.float32).cpu().numpy()  # (hidden_dim,)
            layer_embeddings.append(layer_emb)
        layer_embeddings = np.stack(layer_embeddings, axis=0)  # (num_layers, hidden_dim)
        # Now average across all layers to get a single embedding
        final_embedding = layer_embeddings.mean(axis=0)  # (hidden_dim,)
        print(f"Final averaged embedding shape for {img_file}: {final_embedding.shape}")
        embeddings[img_file] = final_embedding
        np.save(output_path, embeddings)
        print(
            f"Extracted and saved final averaged embedding for {img_file}, shape: {final_embedding.shape}. Total: {len(embeddings)}"
        )
    elif num_im_end_tokens == 0:
        print(f"No <|im_end|> tokens found in {img_file}")
    else:
        print(f"Warning: More than one <|im_end|> token found in {img_file}. Skipping.")
