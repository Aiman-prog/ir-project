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
from nltk.stem import WordNetLemmatizer
import nltk
import requests
import json

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
    nltk.download('punkt')

class SemanticSearchEngineExperimental:
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
        self.embeddings = None
        self.api_key = api_key
        
        # Get stopwords and initialize lemmatizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing - just lowercase and remove special characters."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers, keeping spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
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
        """Create FAISS index from document embeddings."""
        print("Creating search index...")
        total_docs = len(self.documents)
        dimension = self.model.get_sentence_embedding_dimension()
        
        # Preprocess documents
        print("Preprocessing documents...")
        processed_docs = [self.preprocess_text(doc) for doc in self.documents]
        
        # Initialize FAISS index with FlatL2 for stability
        self.index = faiss.IndexFlatL2(dimension)
        
        # Process embeddings in smaller batches
        print("Creating embeddings...")
        all_embeddings = []
        for i in tqdm(range(0, total_docs, batch_size)):
            batch_texts = processed_docs[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
            
            # Add to index in smaller chunks to prevent memory issues
            if len(all_embeddings) >= 2:
                chunk = np.vstack(all_embeddings)
                self.index.add(chunk)
                all_embeddings = []
        
        # Add any remaining embeddings
        if all_embeddings:
            chunk = np.vstack(all_embeddings)
            self.index.add(chunk)
    
    def search(self, query: str, k: int = 5, reformulate: bool = True) -> List[Tuple[str, str, float]]:
        """
        Search for similar documents using semantic similarity.
        
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
        
        # Generate query embedding with normalization
        query_embedding = self.model.encode(
            [processed_query],
            normalize_embeddings=True
        )
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert distances to similarity scores (cosine similarity)
        scores = 1 - distances[0] / 2  # Convert L2 distance to cosine similarity
        
        # Return results with IDs
        results = [(self.document_ids[idx], self.documents[idx], score) 
                  for idx, score in zip(indices[0], scores)]
        return results
    
    def save_index(self, path: str):
        """Save the search index and documents to disk."""
        if self.index is None:
            raise ValueError("No index to save. Call create_index() first.")
            
        print(f"Saving index to {path}...")
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
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
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        
        # Load documents and IDs
        with open(os.path.join(path, 'documents.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_ids = data['document_ids'] 