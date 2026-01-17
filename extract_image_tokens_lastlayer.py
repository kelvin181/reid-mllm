import os
import argparse
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image


# Choose whether to process query images or gallery images (set True for query)
query = False

if query:
    default_out = os.path.join(os.getcwd(), "query_img_tokens_lastlayer.npy")
    default_img_dir = os.path.join(os.getcwd(), "market1501", "market-1501-v15.09.15", "query")
else:
    default_out = os.path.join(os.getcwd(), "gallery_img_tokens_lastlayer.npy")
    default_img_dir = os.path.join(os.getcwd(), "market1501", "market-1501-v15.09.15", "bounding_box_test")
print(f"Image directory: {default_img_dir}")


def find_token_id(tokenizer, candidates, input_ids_np):
    for s in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
        except Exception:
            tid = None
        if isinstance(tid, int):
            if (input_ids_np == tid).any():
                return tid, s
    # fallback to tokenizer attribute if exists
    if hasattr(tokenizer, "image_token_id"):
        tid = getattr(tokenizer, "image_token_id")
        if isinstance(tid, int) and (input_ids_np == tid).any():
            return tid, "tokenizer.image_token_id"
    return None, None


def main(args):
    device = torch.device("cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model).to(device)
    processor = AutoProcessor.from_pretrained(args.model)

    img_dir = args.img_dir
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    print(f"Found {len(img_files)} images in {img_dir}")

    embeddings = {}
    if os.path.exists(out_path):
        embeddings = np.load(out_path, allow_pickle=True).item()

    for img_file in img_files:
        if img_file in embeddings:
            print(f"Skipping {img_file}, already extracted")
            continue
        img_path = os.path.join(img_dir, img_file)
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": args.prompt}
            ]
        }]

        raw_img = Image.open(img_path)
        inputs = processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of layers
        last_layer = hidden_states[-1]  # (1, seq_len, hidden_dim)
        input_ids = inputs['input_ids'][0].cpu().numpy()

        # find im_start and im_end token ids by candidate strings
        im_start_candidates = ["<|im_start|>", "<|image|>", "<|img|>", "<|image_start|>"]
        im_end_candidates = ["<|im_end|>", "<|image_end|>"]

        start_id, start_str = find_token_id(processor.tokenizer, im_start_candidates, input_ids)
        end_id, end_str = find_token_id(processor.tokenizer, im_end_candidates, input_ids)

        # Determine image token positions between start and end (exclusive)
        positions = []
        if start_id is not None and end_id is not None:
            print(f"{img_file}: found start token '{start_str}' and end token '{end_str}'")
            # use first occurrence of start and last occurrence of end
            start_pos = np.where(input_ids == start_id)[0][0]
            end_pos = np.where(input_ids == end_id)[0][-1]
            print(f"{img_file}: number of image tokens: {end_pos - start_pos - 1}")
            if end_pos > start_pos + 1:
                positions = list(range(start_pos + 1, end_pos))
            else:
                positions = []

        # Average last layer hidden states at these positions
        vecs = last_layer[0, positions, :].to(torch.float32)
        avg_vec = vecs.mean(dim=0).cpu().numpy()

        embeddings[img_file] = avg_vec
        np.save(out_path, embeddings)
        print(f"Saved {img_file}: tokens={len(positions)}, emb_shape={avg_vec.shape}. Total={len(embeddings)}")

    print(f"Done. Saved embeddings to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-VL-2B-Instruct')
    p.add_argument('--img_dir', default=default_img_dir)
    p.add_argument('--out', default=default_out)
    p.add_argument('--prompt', default='Summarize the person image, focusing on age, gender, clothing, and biometric features')
    args = p.parse_args()
    main(args)
