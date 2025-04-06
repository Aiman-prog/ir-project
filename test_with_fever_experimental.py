import pandas as pd
import pickle
from semantic_search_engine_bm25 import SemanticSearchEngineBM25
from semantic_search_engine_faiss import SemanticSearchEngineExperimental
from semantic_search_engine_cosine import SemanticSearchEngineCosine
from tqdm import tqdm

def main():
    # Initialize search engine with DeepSeek API key
    api_key = "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"  # Your DeepSeek API key
    search_engine = SemanticSearchEngineCosine(api_key=api_key)
    
    # Load corpus
    print("Loading corpus...")
    corpus_df = pd.read_csv('corpus_df.csv')
    
    # Set up documents and document IDs
    search_engine.documents = corpus_df['text'].tolist()
    search_engine.document_ids = corpus_df['doc_id'].astype(str).tolist()
    
    # Create and save index
    print("Creating index...")
    search_engine.create_index()
    search_engine.save_index('cosine_index')
    
    # Load queries
    print("Loading queries...")
    queries_df = pd.read_csv('filtered_queries_df.csv')
    
    # Load qrels
    print("Loading qrels...")
    with open('qrels2.pkl', 'rb') as f:
        qrels = pickle.load(f)
    
    # Process all queries at once
    total_mrr = 0
    total_precision = 0
    total_recall = 0
    num_queries = 50
    successful_searches = 0
    
    print("\nProcessing all queries and calculating metrics...")
    for idx, row in tqdm(queries_df.iloc[:num_queries].iterrows(), total=num_queries):
        claim = row['claim']
        
        # Get relevant document IDs from qrels
        relevant_doc_ids = set(str(doc_id) for doc_id in qrels.get(claim, []))
        
        # Search for top 5 documents with reformulation enabled
        results = search_engine.search(claim, k=5, reformulate=True)
        
        # Calculate MRR@5
        mrr = 0
        retrieved_relevant = 0
        retrieved_doc_ids = set()
        
        for rank, (doc_id, doc_text, score) in enumerate(results, 1):
            retrieved_doc_ids.add(doc_id)
            if doc_id in relevant_doc_ids:
                retrieved_relevant += 1
                if mrr == 0:  # Only count first relevant doc for MRR
                    mrr = 1.0 / rank
                    successful_searches += 1
                    print(f"\nFound relevant document at rank {rank} for claim: {claim[:100]}...")
                    print(f"Document ID: {doc_id}")
                    print(f"Score: {score:.4f}")
        
        # Calculate precision and recall for this query
        precision = retrieved_relevant / len(results) if results else 0
        recall = retrieved_relevant / len(relevant_doc_ids) if relevant_doc_ids else 0
        
        # Update totals
        total_mrr += mrr
        total_precision += precision
        total_recall += recall
        
        # Print per-query metrics
        print(f"\nMetrics for query {idx + 1}:")
        print(f"MRR@5: {mrr:.4f}")
        print(f"Precision@5: {precision:.4f}")
        print(f"Recall@5: {recall:.4f}")
    
    # Calculate and print final results
    avg_mrr = total_mrr / num_queries
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    
    print(f"\nFinal Results Summary:")
    print(f"Total queries processed: {num_queries}")
    print(f"Successful searches (found at least one relevant document): {successful_searches}")
    print(f"Average MRR@5: {avg_mrr:.4f}")
    print(f"Average Precision@5: {avg_precision:.4f}")
    print(f"Average Recall@5: {avg_recall:.4f}")
    print(f"Success rate: {(successful_searches/num_queries)*100:.2f}%")

if __name__ == "__main__":
    main() 