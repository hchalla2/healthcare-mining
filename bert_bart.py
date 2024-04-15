from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import os
# Load BERT for retrieval
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def process_file_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        url_line = lines[0].strip()
        text = lines[2].strip()

    # Extract URL from the line
    url = url_line.split('url:')[1].strip()
    # Extract title from URL
    title = url.split('/')[-1]
    
    # Construct JSON document
    return text;

def remove_prefix(text):
    
    ans = text.split("Feedback");
    return ans[len(ans)-1]


# Encode documents
def encode_documents(doc_texts):
    encoded = bert_tokenizer(doc_texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        doc_embeddings = bert_model(**encoded).last_hidden_state.mean(dim=1)
    return doc_embeddings

# Query processing and document retrieval (simplified)
def retrieve_documents(query):
    query_encoded = bert_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    query_embedding = bert_model(**query_encoded).last_hidden_state.mean(dim=1)
    # Assuming 'doc_embeddings' and 'doc_texts' are loaded and available
    query_embedding_np = query_embedding.detach().numpy()
    doc_embeddings_np = doc_embeddings.detach().numpy()

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding_np, doc_embeddings_np)
    top_k = 10;
    top_idx = similarities.argsort()[0][-top_k:]  # Get indices of top documents
    return [doc_texts[idx] for idx in top_idx]

# Load BART for generation
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Generate answer from retrieved docs
def generate_answer(documents):
    inputs = bart_tokenizer(documents, return_tensors='pt', truncation=True, padding=True, max_length=1024)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
directory = 'patient-info'
doc_texts = [];
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    text = process_file_to_json(filepath)
    # Remove the prefix
    result_string = remove_prefix(text)
    doc_texts.append(result_string)
   
query = "What are the treatments for breast cancer?"
doc_embeddings = encode_documents(doc_texts)
retrieved_docs = retrieve_documents(query)
prompt = "Question: " + query + " , using the following information " + " ".join(retrieved_docs);
answer = generate_answer(prompt)
print(answer)