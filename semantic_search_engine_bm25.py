import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import faiss
import pickle
import os
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import requests
import json
from rank_bm25 import BM25Okapi
from nltk.stem import WordNetLemmatizer

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

class SemanticSearchEngineBM25:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', api_key: str = None):
        """
        Initialize the semantic search engine with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            api_key (str): DeepSeek API key for query reformulation
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.document_ids = []
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
    
    def create_index(self):
        """Create BM25 index from document tokens."""
        print("Creating search index...")
        
        # Preprocess documents
        print("Preprocessing documents...")
        processed_docs = [self.preprocess_text(doc) for doc in self.documents]
        
        # Tokenize documents
        print("Creating tokenized documents...")
        tokenized_docs = [word_tokenize(text) for text in processed_docs]
        
        # Create BM25 index
        print("Creating BM25 index...")
        self.index = BM25Okapi(tokenized_docs)
        print(f"Index created with {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5, reformulate: bool = True) -> List[Tuple[str, str, float]]:
        """
        Search for similar documents using BM25.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            reformulate (bool): Whether to use DeepSeek for query reformulation
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
            
        # Preprocess and potentially reformulate query
        processed_query = self.preprocess_text(query)
        if reformulate and self.api_key:
            reformulated_query = self.reformulate_query_with_deepseek(processed_query)
            print(f"Original query: {processed_query}")
            print(f"Reformulated query: {reformulated_query}")
            processed_query = reformulated_query
        
        # Tokenize query
        query_tokens = word_tokenize(processed_query)
        query_tokens = [token for token in query_tokens if token not in self.stop_words]
        
        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)
        
        # Get top k documents
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Return results with IDs
        results = [(self.document_ids[idx], self.documents[idx], float(scores[idx])) 
                  for idx in top_k_indices]
        return results
    
    def save_index(self, path: str):
        """Save the search index and documents to disk."""
        if self.index is None:
            raise ValueError("No index to save. Call create_index() first.")
            
        print(f"Saving index to {path}...")
        os.makedirs(path, exist_ok=True)
        
        # Save documents and IDs
        data = {
            'documents': self.documents,
            'document_ids': self.document_ids
        }
        with open(os.path.join(path, 'documents.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
    def load_index(self, path: str):
        """Load the search index and documents from disk."""
        print(f"Loading index from {path}...")
        
        # Load documents and IDs
        with open(os.path.join(path, 'documents.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_ids = data['document_ids']
        
        # Recreate BM25 index
        self.create_index() 