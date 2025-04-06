import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import requests
import json
from sklearn.preprocessing import normalize


# https://huggingface.co/blog/g-ronimo/semscore

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

class SemanticSearchEngineCosine:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', api_key: str = None):
        """
        Initialize the semantic search engine with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            api_key (str): DeepSeek API key for query reformulation
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_ids = []
        self.embeddings = None
        self.api_key = api_key
        
        # Get stopwords and initialize lemmatizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text with basic lemmatization and cleaning."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and process words
        words = word_tokenize(text)
        processed_words = []
        
        for word in words:
            if word not in self.stop_words:
                # Lemmatize the word
                lemma = self.lemmatizer.lemmatize(word)
                processed_words.append(lemma)
        
        return ' '.join(processed_words)
    
    def reformulate_query_with_deepseek(self, query: str) -> str:
        """Optimize query for semantic search using transformer models."""
        if not self.api_key:
            return query
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/aimanmajeed/IR_final",
            "X-Title": "IR Final Project"
        }
        
        prompt = f"""Convert this text into a Wikipedia-style search query where:
- All text is lowercase
- Special characters and numbers are removed
- Words are lemmatized to their base form
- Only letters and spaces remain

Example:
Input: "The COVID-19 vaccine was developed in 2020"
Output: "covid vaccine development timeline history"

{query}

Your turn:"""
        
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a Wikipedia search expert that helps convert queries into clear, factual search terms."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            reformulated_query = response.json()["choices"][0]["message"]["content"].strip().lower()
            reformulated_query = re.sub(r'[^a-z\s]', '', reformulated_query)
            reformulated_query = ' '.join(reformulated_query.split())
            
            print(f"\nOriginal query: {query}")
            print(f"Reformulated query: {reformulated_query}")
            
            return reformulated_query
            
        except Exception as e:
            print(f"Error in query reformulation: {e}")
            return query
    
    def create_index(self, batch_size: int = 500):
        """Create document embeddings for cosine similarity search."""
        total_docs = len(self.documents)
        
        # Preprocess all documents first
        processed_docs = [self.preprocess_text(doc) for doc in self.documents]
        
        # Process documents in batches
        all_embeddings = []
        for i in tqdm(range(0, total_docs, batch_size)):
            batch_texts = processed_docs[i:i + batch_size]
            # Generate embeddings with normalization for cosine similarity
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True,  
                convert_to_tensor=True,
                batch_size=32
            )
            all_embeddings.append(batch_embeddings.cpu().numpy())
            
        # Concatenate all embeddings
        self.embeddings = np.vstack(all_embeddings)
        print(f"Created embeddings for {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5, reformulate: bool = True) -> List[Tuple[str, str, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            reformulate (bool): Whether to use DeepSeek for query reformulation
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Call create_index() first.")
            
        # Preprocess and potentially reformulate query
        processed_query = self.preprocess_text(query)
        if reformulate and self.api_key:
            reformulated_query = self.reformulate_query_with_deepseek(processed_query)
            print(f"Original query: {processed_query}")
            print(f"Reformulated query: {reformulated_query}")
            processed_query = reformulated_query
        
        # Generate query embedding (normalized for cosine similarity)
        query_embedding = self.model.encode(
            [processed_query],
            normalize_embeddings=True, 
            convert_to_tensor=True
        ).cpu().numpy()
        
        # Calculate cosine similarity scores (dot product of normalized vectors)
        scores = np.dot(self.embeddings, query_embedding.T).squeeze()
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        # Return results with IDs
        results = [(self.document_ids[idx], self.documents[idx], float(score)) 
                  for idx, score in zip(top_k_indices, top_k_scores)]
        return results
    
    def save_index(self, path: str):
        """Save the embeddings and documents to disk."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Call create_index() first.")
            
        print(f"Saving embeddings to {path}...")
        os.makedirs(path, exist_ok=True)
        
        # Save documents, IDs and embeddings
        data = {
            'documents': self.documents,
            'document_ids': self.document_ids,
            'embeddings': self.embeddings
        }
        with open(os.path.join(path, 'cosine_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
    def load_index(self, path: str):
        """Load the embeddings and documents from disk."""
        print(f"Loading embeddings from {path}...")
        
        # Load documents, IDs and embeddings
        with open(os.path.join(path, 'cosine_data.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_ids = data['document_ids']
            self.embeddings = data['embeddings']

def main():
    # Initialize search engine with your OpenRouter API key
    api_key = "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"  # Replace with your actual API key
    search_engine = SemanticSearchEngineCosine(api_key=api_key)
    
    # First, load the wiki pages
    search_engine.load_wiki_pages(batch_size=1000)
    
    # Create and save the index
    search_engine.create_index()
    search_engine.save_index('wiki_search_index_cosine')
    
    # Example search with query reformulation
    query = "covid-19 is fake"
    results = search_engine.search(query, k=5, reformulate=True)
    
    print(f"\nSearch results:")
    for doc_id, doc_text, score in results:
        print(f"\nDocument ID: {doc_id}")
        print(f"Score: {score:.4f}")
        print(f"Text: {doc_text[:200]}...")  # Print first 200 characters of text

if __name__ == "__main__":
    main() 