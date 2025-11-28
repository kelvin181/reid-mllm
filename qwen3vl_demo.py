"""
Overview for Qwen3-VL-2B-Instruct demo:

1. User Input:
    - You provide a text prompt and an image to the program.

2. Pre-processing / Normalization:
    - The text is cleaned and prepared (e.g., spaces removed, lowercased).
    - The image is resized, cropped, and converted to a standard format so the model can use it.

3. Tokenization (Text) & Feature Extraction (Image):
    - The text is tokenized and each token is given a number (token ID).
    - The image is turned into a grid of numbers (pixel values) that represent its colors and features.

4. Packing into Model Input Format:
    - The text tokens and image features are combined into a single package.
    - Package contains extra info (like attention masks and special tokens) so the model knows how to read them.

5. Inputs Sent to the Model

6. Embedding Layer:
    - The model turns the token IDs and image features into vectors (embeddings) that it can understand and work with.

7. Neural Network Processing:
    - The model uses these embeddings to think, reason, and generate answers by passing them through many layers of its neural network.

8. Generation Output:
    - The model produces a response based on your prompt and image.

Note:
    - Steps 1-5 are done by the code and the processor.
    - Steps 6 and 7 are done inside the model itself.
"""

import time, os
import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pprint import pprint
from PIL import Image


# Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
print("Model loaded")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-2B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print("processor loaded")

# --- Create messages to LLM with images and text ---
img_dir = r"e:/Projects/Honours/market1501/market-1501-v15.09.15/bounding_box_test"
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
img_files = img_files[:3]

print(len(img_files), "images to process.")
for idx, img_file in enumerate(img_files):
    img_path = os.path.join(img_dir, img_file)
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text",
                    "text": "Summarize the person image, focusing on age, gender, clothing, and biometric features",
                },
            ],
        }
    ]
    print(f"\nProcessing image {idx+1}/{len(img_files)}: {img_file}")
    print("Prompt message:")
    pprint(message)

    # --- Pre-processing / Normalization ---
    # (Text normalization is handled inside processor)
    # For image, show raw image array
    # This section is done by the processor

    raw_img = Image.open(img_path)
    print("\n--- Pre-processing / Normalization ---")
    print("Raw image array shape:", np.array(raw_img).shape)
    print("Raw image array sample:", np.array(raw_img).flatten()[:10])

    # Tokenization (Text) & Feature Extraction (Image)
    text = message[0]["content"][1]["text"]
    tokens = processor.tokenizer.tokenize(text)
    token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
    print("\nTokenization & Feature Extraction")
    print("Text tokens:", tokens)
    print("Text token IDs:", token_ids)

    # Image feature extraction (pixel_values)
    image_features = processor.image_processor(raw_img)
    print("Image features: ", image_features)

    # --- Packing into Model Input Format ---
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    print("\n--- Packed Model Inputs ---")
    for k, v in inputs.items():
        print(f"{k}: shape {v.shape if hasattr(v, 'shape') else type(v)}")
        if hasattr(v, "flatten"):
            print(f"Sample of {k}: {v.flatten()[:10].tolist()}")

    # --- Embedding Layer (Inside the Model) ---
    # Converts raw inputs (tokens, pixels) into vectors the model can work with
    with torch.no_grad(): # Disable gradient calc, save mem, speed up computation, since we only want to see hidden states
        outputs = model(**inputs, output_hidden_states=True)
    print("\n--- Embedding Layer Output ---")
    if hasattr(outputs, "hidden_states"):
        emb = outputs.hidden_states
        print("Embedding layer output (first hidden state) shape:", emb[0].shape)
        print("Embedding layer output sample:", emb[0].flatten()[:10].tolist())
    else:
        print("No hidden_states found in model outputs.")

    # --- Neural Network Processing ---
    # NN refines the vectors and adds more meaning and context to them
    # Final layer is what the model uses to generate predictions
    if hasattr(outputs, "hidden_states"):
        print("Final hidden state (last layer) shape:", emb[-1].shape)
        print("Final hidden state sample:", emb[-1].flatten()[:10].tolist())
    else:
        print("No hidden_states found in model outputs.")

    # --- Generation Output ---
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n--- Generation Output ---\nGeneration time: {elapsed:.2f} seconds")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("Output text:")
    pprint(output_text)
