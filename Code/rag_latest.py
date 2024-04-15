import os
import json
from sentence_transformers import SentenceTransformer
from torch import embedding
import numpy as np
from datasets import Dataset, Features, Value, Sequence, Array2D
import torch
from transformers import BertTokenizer, RagTokenizer

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, BartTokenizer, RagTokenizer, RagTokenForGeneration

# For DPR component
dpr_question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

from transformers import MaxLengthCriteria, StoppingCriteriaList

stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=1000)])

def get_embeddings(texts):
    model = SentenceTransformer('all-mpnet-base-v2')
    return np.array([model.encode(text) for text in texts])

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
    json_document = {
        "title": title,
        "text": text
    }

    return json_document


def load_documents(directory):
    documents = {}
    documents["text"] = [];
    documents["title"] = [];
    documents["embeddings"] = [];
    cnt = 0;
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        json_output = process_file_to_json(path)
        documents["text"].append(json_output["text"])
        documents["title"].append(json_output["title"])
        if cnt%3==0:
            print(cnt)
        if cnt>=100:
            break;
        cnt = cnt + 1;
    return documents

directory = "patient-info/"
documents = load_documents(directory)

embeddings = get_embeddings(documents["text"])
# Ensure embeddings are 2D if they are not already
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(-1, 768)

documents["embeddings"] = embeddings;
print("Shape: ")
print(embeddings.shape)

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import Dataset

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

# Simulating a dataset
data = documents
dataset = Dataset.from_dict(data)

# Adding FAISS index for the 'embeddings' column
from datasets import Features, Value, Array2D
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Define the dataset features with the correct types
features = Features({
    'title': Value('string'),
    'text': Value('string'),
    'embeddings': Sequence(feature=Value('float32'), length=768)
})

def get_question_hidden_states(question, model, tokenizer):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # Perform encoding using the question encoder
    outputs = model.rag.question_encoder(input_ids)

    #print(type(outputs))
    #print(outputs)
    # Correctly accessing the pooler output
    question_hidden_states = outputs[0]

    return question_hidden_states, input_ids


dataset = Dataset.from_dict(data, features=features)

# Add FAISS index
dataset.add_faiss_index(column='embeddings')

# Initialize retriever with dummy parameters (assuming documents are in dataset)
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", indexed_dataset=dataset)

import torch
from torch import nn

def calculate_cosine_similarity(query_embedding, doc_embeddings):
    # Ensure input is a PyTorch tensor
    if not isinstance(doc_embeddings, torch.Tensor):
        doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float32)
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

    # Normalize the query and document embeddings to unit vectors
    query_norm = query_embedding / (query_embedding.norm(dim=1, keepdim=True) + 1e-10)
    doc_norm = doc_embeddings / (doc_embeddings.norm(dim=2, keepdim=True) + 1e-10)

    # Calculate cosine similarity
    # Adjusted for 3D doc_embeddings: assuming size [batch_size, num_docs, embedding_size]
    cosine_sim = torch.bmm(doc_norm, query_norm.unsqueeze(2)).squeeze(2)  # bmm for batch matmul, adjust dimensions as necessary

    return cosine_sim


def answer_query(query):
    question_hidden_states, input_ids = get_question_hidden_states(query, model, tokenizer)

    if isinstance(question_hidden_states, torch.Tensor):
        question_hidden_states = question_hidden_states.detach().cpu().numpy()  # Ensure it's on CPU and detached

    
    # Temporarily capture the retriever output to inspect it
    retriever_output = retriever(input_ids, question_hidden_states, n_docs=5)
    print("Retriever output:", retriever_output)
    
    context_input_ids = retriever_output["context_input_ids"]
    context_attention_mask = retriever_output["context_attention_mask"]
    
    print(retriever_output.keys())

    # Ensure context_input_ids and context_attention_mask are tensors before passing to generate
    if not isinstance(context_input_ids, torch.Tensor):
        context_input_ids = torch.tensor(context_input_ids)
    if not isinstance(context_attention_mask, torch.Tensor):
        context_attention_mask = torch.tensor(context_attention_mask)
    
    inputs = tokenizer(query, return_tensors='pt')
    query_embedding = dpr_question_encoder(**inputs).pooler_output

    print("Input ids shape: ", input_ids.shape)
    print("Context_input_ids shape ", context_input_ids.shape)
    print("context_attention_mask shape ", context_attention_mask.shape)

    print("Before generate call:")
    print("Input IDs:", input_ids)
    print("Context Input IDs:", context_input_ids)
    print("Context Attention Mask:", context_attention_mask)

    if context_input_ids is None:
        raise ValueError("context_input_ids is unexpectedly None before calling generate.")
    
    doc_scores = calculate_cosine_similarity(query_embedding, retriever_output['retrieved_doc_embeds'])

    # Continue with your existing logic
    outputs = model.generate(
        input_ids=input_ids,
        context_input_ids=context_input_ids,
        context_attention_mask=context_attention_mask,
        doc_scores = doc_scores,
        num_beams=5,
        num_return_sequences=1,
        max_length=600, # Increase max_length
        min_length=100,
        temperature=0.7,  # Example: lowering temperature
        repetition_penalty=1.5,  # Discourages repetition
        stopping_criteria=stopping_criteria,
        top_k=50,
        top_p=0.95
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


query = "what are breast cancer treatments ?";
response = answer_query(query)
print(response)
