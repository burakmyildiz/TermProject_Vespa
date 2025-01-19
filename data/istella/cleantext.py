import re
import json
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_document_fields(document):
    for field in ['title', 'url', 'text', 'extra_text']:
        if field in document:
            document[field] = clean_text(document[field])
    return document

def count_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def process_and_save_documents(input_file_path, output_file_path):

    try:
        total_lines = count_lines(input_file_path)
        if total_lines == 0:
            print(f"Error: Input file not found or empty at {input_file_path}")
            return

        documents = []
        print("Reading and parsing documents...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Reading"):
                try:
                    doc = json.loads(line.strip())
                    if isinstance(doc, dict):  
                        documents.append(doc)
                except json.JSONDecodeError:
                    continue  

        print("\nCleaning documents...")
        cleaned_documents = []
        for doc in tqdm(documents, desc="Cleaning"):
            cleaned_doc = clean_document_fields(doc.copy())
            cleaned_documents.append(cleaned_doc)

        print("\nSaving cleaned documents...")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for doc in tqdm(cleaned_documents, desc="Saving"):
                json_line = json.dumps(doc, ensure_ascii=False)
                outfile.write(json_line + '\n')

        print(f"\nProcess completed successfully!")
        print(f"Total documents processed: {len(cleaned_documents)}")
        print(f"Output saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
    except IOError as e:
        print(f"Error processing files: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    input_file = 'documents_with_embeddings.jsonl' 
    output_file = 'cleaned_documents_with_embeddings.jsonl' 
    process_and_save_documents(input_file, output_file)
