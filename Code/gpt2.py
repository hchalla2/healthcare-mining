import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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




class SimpleRetriever:
    def __init__(self, directory):
        self.documents = []
        self.doc_names = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.load_documents(directory)
    
    def load_documents(self, directory):
        """ Load and store documents from a specified directory. """
        flag = 0;
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            text = process_file_to_json(filepath)
            # Remove the prefix
            result_string = remove_prefix(text)
            if flag==0:
                print("Result string ", result_string)
                flag = 1;
            self.documents.append(result_string)
            self.doc_names.append(filename)
        
        # Create TF-IDF model
        self.document_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
    
    def retrieve(self, query, top_n=10):
        """ Retrieve top N documents based on the query. """
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vec, self.document_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
        return [(self.doc_names[index], self.documents[index]) for index in related_docs_indices]

# Usage:
# directory_path = 'path_to_your_document_directory'
# retriever = SimpleRetriever(directory_path)
# top_documents = retriever.retrieve("Your query here", top_n=10)

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class DocumentResponder:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def generate_response(self, documents):
        combined_text = " ".join(documents)
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=100, pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Check if inputs are within acceptable range
        if input_ids.nelement() == 0 or max(input_ids[0]) >= self.tokenizer.vocab_size:
            raise ValueError("Input token indices are out of range.")

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=101,  # Can be adjusted
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage:
# responder = DocumentResponder()
# response_text = responder.generate_response([doc[1] for doc in top_documents])

# Define directory path and query
directory_path = 'patient-info'
#query = "What are the various types of cancer in world ?"

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/greet', methods=['POST'])
def greet():
    data = request.json
    name = data.get('question')
    greeting = f"Hello {name}"
    query = name
    # Retrieve top documents
    retriever = SimpleRetriever(directory_path)
    top_documents = retriever.retrieve(query, top_n=10)

    print(top_documents)
    # Generate response from documents
    responder = DocumentResponder()
    response_text = responder.generate_response([doc[1] for doc in top_documents])

    print("Generated Response:", response_text)
    return jsonify({'greeting': response_text})

if __name__ == '__main__':
    app.run(debug=True)


