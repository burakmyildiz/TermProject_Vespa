#!/usr/bin/env python

import json
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from contextlib import nullcontext

def remove_illegal_chars(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', s)

def count_lines(file_path):

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for _ in f)

def generate_embeddings(
    input_file,
    output_file,
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    batch_size=32, 
    device=None
):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    #if fp16 and device == "cuda":
    #   model.half()  # FP16

    total_docs = count_lines(input_file)
    print(f"Total documents to process: {total_docs}")

    with open(input_file, 'r', encoding='utf-8', errors='replace') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        batch_texts = []
        batch_docs = []
        pbar = tqdm(total=total_docs, desc="Generating Embeddings", unit="doc")

        for line_num, line in enumerate(fin, 1):
            try:
                doc = json.loads(line)

                title = doc.get("title", "")
                text = doc.get("text", "")
                extra_text = doc.get("extra_text", "")

                combined_text = " ".join(filter(None, [str(title), str(text), str(extra_text)]))
                #combined_text = " ".join(filter(None, [str(title)]))
                combined_text = remove_illegal_chars(combined_text)

                batch_texts.append(combined_text)
                batch_docs.append(doc)

                if len(batch_texts) >= batch_size:
                    with torch.amp.autocast('cuda') if device == "cuda" else nullcontext():
                        embeddings = model.encode(
                            batch_texts,
                            batch_size=batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True  
                        )


                    for batch_doc, embedding in zip(batch_docs, embeddings):
                        batch_doc["embedding"] = embedding.tolist()
                        fout.write(json.dumps(batch_doc, ensure_ascii=False) + "\n")


                    batch_texts = []
                    batch_docs = []
                    pbar.update(batch_size)


                    if device == "cuda":
                        torch.cuda.empty_cache()

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line {line_num}")
                pbar.update(1)
                continue
            except Exception as e:
                print(f"Error processing document on line {line_num}: {e}")
                pbar.update(1)
                continue


        if batch_texts:
            with torch.amp.autocast('cuda') if device == "cuda" else nullcontext():
                embeddings = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )


            for batch_doc, embedding in zip(batch_docs, embeddings):
                batch_doc["embedding"] = embedding.tolist()
                fout.write(json.dumps(batch_doc, ensure_ascii=False) + "\n")

            pbar.update(len(batch_texts))

        pbar.close()

    print(f"Embedding generation completed. Output saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and add SBERT embeddings to JSONL documents without altering original fields.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSONL file with embeddings")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="SBERT model name")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")

    args = parser.parse_args()

    generate_embeddings(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device
    )
