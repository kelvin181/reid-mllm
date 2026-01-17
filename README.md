# ReID-Mllm: Person Re-Identification with Multimodal Large Language Model Embeddings

## Overview
This project extracts image embeddings from the Qwen3VL multimodal model for the Market-1501 dataset and evaluates person re-identification (ReID) performance using CMC and mAP metrics.

## Development
Create a Python virtual environment for development:
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install torch numpy scipy transformers pillow
```

## Generate Embeddings
To generate embeddings, open the generation Python file you want to use (e.g., `generate_embeddings_avg.py`).

Set the `query` variable at the top of the file:
- Set `query = True` to generate query embeddings
- Set `query = False` to generate gallery embeddings

Then run the script:
```
python generate_embeddings_avg.py
```
The script will use the appropriate default input and output paths based on the value of `query`.

## Evaluate ReID Performance
Set the embeddings path to the correct paths at the bottom of `evaluate.py` (e.g, for last layer embeddings).
```
query_emb_path = os.path.join(os.getcwd(), "query_img_tokens_lastlayer.npy")
gallery_emb_path = os.path.join(os.getcwd(), "gallery_img_tokens_lastlayer.npy")
```

Then run the evaluation script:
```
python evaluate.py
```
This will print mAP, CMC scores, and the top-5 most similar gallery images for each query.
