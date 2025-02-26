1. `extract_text_from_file(file_path)`:
    - Input: Path to a file (HTML, PDF, TXT)
    - Output: Extracted text content as a string
    - Purpose: Handles text extraction from different file types, using BeautifulSoup for HTML and PyPDF2 for PDFs

2. `load_documents(base_dir)`:
    - Input: Base directory containing document subdirectories
    - Output:
        - `documents`: Dictionary with document metadata
        - `doc_paths`: Dictionary mapping document IDs to file paths
    - Purpose: Walks through directories, extracts text from files, and creates a structured document collection

3. `preprocess_text(text)`:
    - Input: Raw text string
    - Output: List of preprocessed tokens
    - Processing steps:
        - Convert to lowercase
        - Tokenize
        - Remove punctuation
        - Remove stopwords
        - Apply stemming

4. `preprocess_documents(documents)`:
    - Input: Dictionary of documents
    - Output: Updated documents with preprocessed tokens
    - Purpose: Applies `preprocess_text()` to each document

5. `build_inverted_index(documents)`:
    - Input: Dictionary of documents
    - Output: Inverted index (dictionary mapping terms to document IDs)
    - Purpose: Create a reverse lookup of which documents contain specific terms

6. `calculate_term_frequencies(documents)`:
    - Input: Dictionary of documents
    - Output: Documents updated with term frequency counters
    - Purpose: Count how many times each term appears in a document

7. `calculate_tfidf(documents, inverted_index)`:
    - Input: Documents and inverted index
    - Output:
        - TF-IDF scores for terms in documents
        - Document vectors for similarity calculations
    - Purpose: Calculate term importance using TF-IDF with length normalization

8. `calculate_similarity_matrix(doc_vectors)`:
    - Input: Document vectors
    - Output: Similarity matrix between documents
    - Purpose: Compute cosine similarity between all document pairs

9. `search(query, documents, inverted_index, doc_vectors, tolerance=0.8)`:
    - Input:
        - Search query
        - Documents
        - Inverted index
        - Document vectors
    - Output: Ranked list of matching documents with similarity scores
    - Purpose: Find and rank documents relevant to the query

10. `evaluate_search(test_queries, documents, inverted_index, doc_vectors)`:
    - Input:
        - Test queries with relevance judgments
        - Documents
        - Inverted index
        - Document vectors
    - Output: Performance metrics (precision, recall, F1 score)
    - Purpose: Assess search system performance

Code Flow from `main()`:
1. `load_documents()` - Load and extract text from files
2. `preprocess_documents()` - Preprocess all document texts
3. `calculate_term_frequencies()` - Count term occurrences
4. `build_inverted_index()` - Create term-to-document mapping
5. `calculate_tfidf()` - Compute term importance
6. `calculate_similarity_matrix()` - Compute document similarities
7. Test search functionality:
    - `search()` with sample queries
    - `display_search_results()`
8. Performance evaluation:
    - `evaluate_search()` with predefined test queries
    - `display_evaluation_results()`

The code follows a pipeline:
Text Extraction → Preprocessing → Indexing → TF-IDF → Similarity Calculation → Search → Evaluation

Each function builds upon the previous one, transforming the data step by step to create a functional content-based recommender system.
