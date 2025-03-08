import os
import re
import math
import nltk
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import PyPDF2
from difflib import get_close_matches
import matplotlib.pyplot as plt
import numpy as np

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_text_from_file(file_path):
    """
    Extract text from HTML, PDF, or TXT files.
    """
    try:
        # Check file type and handle accordingly
        if file_path.lower().endswith('.html'):
            # Read HTML content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            # Parse HTML to extract clean text
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        elif file_path.lower().endswith('.pdf'):
            # PDF parsing using PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                # Extract text page by page
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "

        else:  # For TXT files and others
            # Directly read text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()

        # Remove extra whitespace and clean text
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def load_documents(base_dir):
    """
    Load all documents from the specified directory structure
    """
    documents = {}
    doc_id = 0

    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        category = os.path.basename(root)

        # Skip if it's the base directory itself
        if category == os.path.basename(base_dir):
            continue

        for file in files:
            # Only process HTML, PDF, and text files
            if file.endswith(('.html', '.txt', '.pdf')):
                file_path = os.path.join(root, file)

                # Extract text
                text = extract_text_from_file(file_path)

                # Skip if no text was extracted
                if not text:
                    continue

                # Store document info
                doc_name = f"{category}/{file}"
                documents[doc_id] = {
                    'id': doc_id,
                    'name': doc_name,
                    'category': category,
                    'path': file_path,
                    'text': text,
                    'tokens': None,  # Will be populated during preprocessing
                    'term_freq': None,  # Will be populated during TF-IDF calculation
                }
                doc_id += 1

    print(f"Loaded {len(documents)} documents from {base_dir}")

    # Print a sample of documents with IDs
    print("\nSample of loaded documents:")
    sample_count = min(5, len(documents))
    for i, (doc_id, doc) in enumerate(list(documents.items())[:sample_count]):
        print(f"Document [{doc_id}]: {doc['name']} ({doc['category']})")

    return documents

def preprocess_text(text):
    """
    Preprocess text: tokenize, remove stopwords, punctuation, and stem
    """
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def preprocess_documents(documents):
    """
    Preprocess all documents in the collection
    """
    for doc_id, doc in documents.items():
        doc['tokens'] = preprocess_text(doc['text'])

    # Display sample of preprocessed documents
    print("\nSample of preprocessed documents:")
    sample_count = min(3, len(documents))
    for i, (doc_id, doc) in enumerate(list(documents.items())[:sample_count]):
        print(f"Document [{doc_id}]: {doc['name']}")
        token_sample = doc['tokens'][:10]
        print(f"  Sample tokens: {token_sample}...")
        print(f"  Total tokens: {len(doc['tokens'])}")

    return documents

def build_inverted_index(documents):
    """
    Build an inverted index mapping terms to documents
    """
    inverted_index = defaultdict(list)

    for doc_id, doc in documents.items():
        # Get unique terms in the document
        unique_terms = set(doc['tokens'])

        # Add document to the posting list of each term
        for term in unique_terms:
            inverted_index[term].append(doc_id)

    # Convert to regular dict and sort the terms alphabetically
    inverted_index = dict(sorted(inverted_index.items()))

    # Display the complete inverted index
    print("\nInverted Index (complete):")
    print("=" * 70)
    print("{:<20} {:<10} {:<30}".format("Term", "Doc Count", "Documents"))
    print("-" * 70)

    for term, doc_ids in inverted_index.items():
        print("{:<20} {:<10} {:<30}".format(
            term, len(doc_ids),
            str([f"[{doc_id}]" for doc_id in doc_ids[:5]]) + "..." if len(doc_ids) > 5 else str([f"[{doc_id}]" for doc_id in doc_ids])
        ))

    return inverted_index

def calculate_term_frequencies(documents):
    """
    Calculate term frequencies for each document
    """
    for doc_id, doc in documents.items():
        # Count term frequencies
        term_freq = Counter(doc['tokens'])
        doc['term_freq'] = term_freq

    # Display term frequencies for a sample document
    print("\nTerm Frequencies for a sample document:")
    if documents:
        sample_doc_id = list(documents.keys())[0]
        doc = documents[sample_doc_id]
        print(f"Document [{sample_doc_id}]: {doc['name']}")
        top_terms = doc['term_freq'].most_common(10)
        print("Top 10 terms by frequency:")
        for term, freq in top_terms:
            print(f"  {term}: {freq}")

    return documents

def calculate_tfidf(documents, inverted_index):
    """
    Calculate TF-IDF scores for all terms in all documents with length normalization
    """
    N = len(documents)  # Total number of documents

    # Calculate IDF for each term
    idf = {}
    for term, doc_ids in inverted_index.items():
        idf[term] = math.log10(N / len(doc_ids))

    # Calculate TF-IDF for each term in each document
    tfidf = {}
    doc_vectors = {}

    for doc_id, doc in documents.items():
        tfidf[doc_id] = {}
        vector = {}

        # Get document length (total number of terms)
        doc_length = len(doc['tokens'])

        # Calculate TF-IDF for each term in the document
        for term, freq in doc['term_freq'].items():
            # Normalized TF (term frequency / document length)
            normalized_tf = freq / doc_length

            # TF-IDF score
            tfidf[doc_id][term] = normalized_tf * idf.get(term, 0)
            vector[term] = tfidf[doc_id][term]

        # Store the document vector (length normalized)
        doc_vectors[doc_id] = vector

    # Display TF-IDF scores for a sample document
    print("\nTF-IDF Scores for a sample document:")
    if documents:
        sample_doc_id = list(documents.keys())[0]
        doc = documents[sample_doc_id]
        print(f"Document [{sample_doc_id}]: {doc['name']}")

        # Get top terms by TF-IDF score
        top_tfidf = sorted(tfidf[sample_doc_id].items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 terms by TF-IDF score:")
        for term, score in top_tfidf:
            print(f"  {term}: {score:.4f}")

    return tfidf, doc_vectors

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two document vectors
    """
    # Find common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())

    # Calculate dot product for common terms
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

    # Calculate magnitudes (Euclidean norms)
    mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

    # Calculate cosine similarity
    if mag1 * mag2 == 0:
        return 0
    else:
        return dot_product / (mag1 * mag2)

def calculate_profile_similarity(user_profile, doc_id, documents):
    """
    Calculate similarity between a user profile and a document
    with amplified personalization effect
    """
    document = documents[doc_id]
    interest_vector = user_profile['interest_vector']

    # If interest vector is empty, return neutral score
    if not interest_vector:
        return 0.5

    # Get document terms set
    doc_terms = set(document['tokens'])

    # Calculate dot product between interest vector and document tokens
    dot_product = 0

    # Amplification factor - higher values create stronger personalization effect
    amplification_factor = 2.5  # More aggressive amplification

    for term, weight in interest_vector.items():
        if term in doc_terms:
            # Apply amplification to matched terms based on their weight
            dot_product += weight * amplification_factor

    # Normalize by document length and interest vector length
    doc_length = len(doc_terms)
    interest_length = math.sqrt(sum(w**2 for w in interest_vector.values()))

    # Calculate base similarity score
    if doc_length > 0 and interest_length > 0:
        similarity = dot_product / (interest_length * math.sqrt(doc_length))
    else:
        similarity = 0

    # Apply stronger category boost if document category matches user interests
    category_boost = 0
    if document['category'] in user_profile['top_categories']:
        # Apply a stronger boost for top categories
        category_rank = user_profile['top_categories'].index(document['category'])
        category_boost = 0.5 - (0.1 * category_rank)  # Higher boost for higher-ranked categories

    # Cap the final score at 1.0
    return min(similarity + category_boost, 1.0)

def get_similar_documents(doc_id, doc_vectors, documents, top_n=5):
    """
    Get top-N similar documents to the given document
    """
    target_vector = doc_vectors[doc_id]
    similarities = []

    # Calculate similarity with all other documents
    for other_id, other_vector in doc_vectors.items():
        if other_id != doc_id:
            sim = cosine_similarity(target_vector, other_vector)
            similarities.append((other_id, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Display the most similar documents
    print(f"\nTop {top_n} most similar documents to [{doc_id}]:")
    print("-" * 70)

    for i, (similar_id, sim) in enumerate(similarities[:top_n]):
        print(f"{i+1}. Document [{similar_id}]: {documents[similar_id]['name']} - Similarity: {sim:.4f}")

    return similarities[:top_n]

def search(query, documents, inverted_index, doc_vectors, user_id=None, user_profiles=None, tolerance=0.8):
    """
    Search for documents matching a query, with support for tolerant retrieval and personalization
    """
    # Preprocess the query
    query_tokens = preprocess_text(query)

    # If no valid tokens after preprocessing, return empty result
    if not query_tokens:
        return []

    # Find matching documents for each query term
    matching_docs = set()  # Initialize the matching_docs set

    for query_term in query_tokens:
        # Try exact matching first
        if query_term in inverted_index:
            matching_docs.update(inverted_index[query_term])
        else:
            # Try fuzzy matching if exact match not found
            all_terms = list(inverted_index.keys())
            close_matches = get_close_matches(query_term, all_terms, n=3, cutoff=tolerance)

            for match in close_matches:
                matching_docs.update(inverted_index[match])

    # If no matching documents found, return empty result
    if not matching_docs:
        return []

    # Calculate query vector
    query_vector = {}
    for term in query_tokens:
        # Use TF-IDF weight if the term is in the corpus, otherwise give it a default weight
        query_vector[term] = query_vector.get(term, 0) + 1

    # Normalize query vector
    query_length = len(query_tokens)
    for term in query_vector:
        query_vector[term] /= query_length

    # Calculate similarity to query for each matching document
    similarities = []

    for doc_id in matching_docs:
        doc_vector = doc_vectors[doc_id]

        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector, doc_vector)

        # Apply personalization if user profile is provided
        if user_id and user_profiles and user_id in user_profiles:
            profile_sim = calculate_profile_similarity(user_profiles[user_id], doc_id, documents)

            # Stronger personalization weight (50%)
            similarity = (0.5 * similarity) + (0.5 * profile_sim)

            # Further amplify based on document category
            doc_category = documents[doc_id]['category']

            # Check if this category is in the user's top categories
            if doc_category in user_profiles[user_id]['top_categories']:
                # Apply a boost based on the category rank (higher for top categories)
                rank = user_profiles[user_id]['top_categories'].index(doc_category)
                rank_boost = 1.3 - (rank * 0.1)  # 1.3, 1.2, 1.1, etc.
                similarity *= rank_boost

        similarities.append((doc_id, similarity))

    # Sort by similarity score (descending)
    ranked_results = sorted(similarities, key=lambda x: x[1], reverse=True)

    return ranked_results

def display_search_results(results, documents, top_n=5):
    """
    Display search results
    """
    if not results:
        print("No matching documents found.")
        return

    print(f"\nTop {min(top_n, len(results))} matching documents:")
    print("-" * 70)

    for i, (doc_id, score) in enumerate(results[:top_n]):
        doc = documents[doc_id]
        print(f"{i + 1}. [{doc_id}] {doc['category']}/{doc['name']} (Score: {score:.4f})")

        # Show snippet of text
        text_preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
        print(f"   {text_preview}\n")

def extract_categories_from_dataset(base_dir):
    """
    Extract category names from directory structure
    """
    categories = []

    # Walk through the directory structure
    for root, dirs, _ in os.walk(base_dir):
        # Get only the immediate subdirectories of the base directory
        if root == base_dir:
            categories = dirs
            break

    print(f"\nExtracted categories from dataset: {categories}")
    return categories

def create_user_profiles(documents, num_profiles=2):
    """
    Create sample user profiles based on document categories
    """
    # Extract all unique categories from documents
    categories = list(set(doc['category'] for doc in documents.values()))

    if len(categories) < 2:
        print("Warning: Not enough categories to create distinct user profiles")
        categories = categories * 2  # Duplicate if needed

    # Create profiles with distinct interests
    user_profiles = {}

    # Create profiles with opposing interests
    for i in range(num_profiles):
        # Sort categories to ensure deterministic behavior
        sorted_categories = sorted(categories)

        # Assign primary interests based on index
        primary_categories = sorted_categories[i::num_profiles]  # Take every num_profiles-th category starting from i

        # Create placeholder for search history
        search_history = []

        # Generate search queries based on primary categories
        for category in primary_categories:
            # Find documents in this category
            category_docs = [doc for doc in documents.values() if doc['category'] == category]

            # Generate search queries from document content
            for doc in category_docs[:3]:  # Use first 3 docs from each category
                # Extract most common terms
                if doc['term_freq']:
                    common_terms = [term for term, _ in doc['term_freq'].most_common(5)]
                    if common_terms:
                        # Create query using 2-3 common terms
                        query_terms = common_terms[:min(3, len(common_terms))]
                        search_history.append(" ".join(query_terms))

        # Limit search history to 10 items
        search_history = search_history[:10]

        # Create user profile
        user_id = f"user_{i+1}"
        user_profiles[user_id] = {
            "interests": primary_categories,
            "top_categories": primary_categories,  # Store categories in order of preference
            "search_history": search_history,
            "interest_vector": {}  # Will be populated below
        }

    # Build interest vectors for each user
    for user_id, profile in user_profiles.items():
        # Create a weighted vector of interests
        interest_vector = Counter()

        # Add terms from interests with higher weight
        for interest in profile["interests"]:
            terms = preprocess_text(interest)
            for term in terms:
                interest_vector[term] += 3  # Higher weight for explicit interests

        # Add terms from search history with lower weight
        for i, query in enumerate(profile["search_history"]):
            terms = preprocess_text(query)
            for term in terms:
                # Weight decreases for less recent searches
                recency_weight = 2.0 * (1.0 - (i / len(profile["search_history"])))
                interest_vector[term] += recency_weight

        # Store the interest vector in the profile
        profile["interest_vector"] = dict(interest_vector)

    # Print user profiles
    print("\nCreated User Profiles:")
    for user_id, profile in user_profiles.items():
        print(f"\n{user_id}:")
        print(f"  Primary interests: {profile['interests']}")
        print(f"  Top categories (in order): {profile['top_categories']}")
        print(f"  Search history (top 5 of {len(profile['search_history'])}):")
        for i, query in enumerate(profile['search_history'][:5]):
            print(f"    - {query}")

        # Show top terms in interest vector
        top_terms = sorted(profile['interest_vector'].items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top terms in interest vector:")
        for term, weight in top_terms:
            print(f"    - {term}: {weight:.2f}")

    return user_profiles

def determine_relevance(doc_id, query_info, documents):
    """
    Determine document relevance based on categories and content

    Args:
        doc_id: Document ID
        query_info: Query information including topics
        documents: Document collection

    Returns:
        float: Relevance score (0.0 to 1.0)
    """
    doc = documents[doc_id]
    doc_category = doc['category']
    doc_text = doc['text'].lower()

    # Check if document category matches any query topics
    category_match = doc_category in query_info['topics']

    # Calculate text match based on query terms
    query_terms = query_info['query'].lower().split()
    term_matches = sum(1 for term in query_terms if term in doc_text)
    term_score = min(term_matches / len(query_terms), 1.0) if query_terms else 0

    # Combine category and term relevance
    if category_match:
        return 0.7 + 0.3 * term_score
    else:
        return 0.4 * term_score

def create_test_queries(documents):
    """
    Create a diverse set of test queries based on document content
    """
    test_queries = []
    categories = set(doc['category'] for doc in documents.values())

    # Create one query per category
    for category in categories:
        # Find documents in this category
        category_docs = [doc for doc in documents.values() if doc['category'] == category]

        if category_docs:
            # Select a random document
            doc = category_docs[0]

            # Extract common terms
            if doc['term_freq']:
                common_terms = [term for term, _ in doc['term_freq'].most_common(5)]
                if common_terms:
                    # Create query using 2-3 common terms
                    query_terms = common_terms[:min(3, len(common_terms))]
                    query = " ".join(query_terms)

                    # Add to test queries
                    test_queries.append({
                        'query': query,
                        'category': category,
                        'topics': [category],
                        'description': f"{category} query"
                    })

    print("\nCreated test queries based on document content:")
    for i, query in enumerate(test_queries):
        print(f"{i+1}. Query: '{query['query']}' (Category: {query['category']})")

    return test_queries

def evaluate_search(test_queries, documents, inverted_index, doc_vectors):
    """
    Evaluate search performance on test queries
    """
    results = {}
    all_aps = []  # For calculating MAP
    precision_at_k_values = [1, 5, 10]  # k values for precision@k
    precision_at_k_results = {k: [] for k in precision_at_k_values}

    print("\nEvaluating Search Performance:")
    print("=" * 70)

    for i, query_info in enumerate(test_queries):
        query = query_info['query']
        target_category = query_info['category']

        print(f"\nEvaluating Query {i+1}: '{query}' (Expected category: {target_category})")

        # Get search results
        search_results = search(query, documents, inverted_index, doc_vectors)

        # Evaluate precision and category relevance
        if not search_results:
            print("  No results found for this query.")
            results[i] = {
                'query': query,
                'precision': 0,
                'category_match': 0,
                'avg_score': 0,
                'num_results': 0,
                'average_precision': 0,
                'precision_at_k': {k: 0 for k in precision_at_k_values}
            }
            all_aps.append(0)
            for k in precision_at_k_values:
                precision_at_k_results[k].append(0)
            continue

        # Calculate metrics
        top_results = search_results
        num_results = len(top_results)

        # Category match (how many results match the target category)
        category_matches = sum(1 for doc_id, _ in top_results
                               if documents[doc_id]['category'] == target_category)
        category_precision = category_matches / num_results if num_results > 0 else 0

        # Average score
        avg_score = sum(score for _, score in top_results) / num_results if num_results > 0 else 0

        # Calculate average precision
        ap = calculate_average_precision(search_results, target_category, documents)
        all_aps.append(ap)

        # Calculate precision@k
        p_at_k = {}
        for k in precision_at_k_values:
            if k <= len(search_results):
                k_matches = sum(1 for doc_id, _ in search_results[:k]
                                if documents[doc_id]['category'] == target_category)
                p_at_k[k] = k_matches / k
                precision_at_k_results[k].append(p_at_k[k])
            else:
                p_at_k[k] = 0 if k > 0 else 1  # Precision@0 is defined as 1
                precision_at_k_results[k].append(p_at_k[k])

        # Store results
        results[i] = {
            'query': query,
            'precision': category_precision,
            'category_match': category_matches,
            'avg_score': avg_score,
            'num_results': num_results,
            'average_precision': ap,
            'precision_at_k': p_at_k
        }

        # Display top 5 results
        display_search_results(search_results, documents, 5)

        # Display metrics
        print(f"  Metrics:")
        print(f"    - Results matching target category: {category_matches}/{num_results} ({category_precision:.2%})")
        print(f"    - Average precision: {ap:.4f}")
        print(f"    - Average relevance score: {avg_score:.4f}")
        for k in precision_at_k_values:
            if k in p_at_k:
                print(f"    - Precision@{k}: {p_at_k[k]:.4f}")

    # Calculate Mean Average Precision (MAP)
    map_score = sum(all_aps) / len(all_aps) if all_aps else 0
    print(f"\nMean Average Precision (MAP): {map_score:.4f}")

    # Calculate mean precision@k
    mean_precision_at_k = {}
    for k in precision_at_k_values:
        if precision_at_k_results[k]:
            mean_precision_at_k[k] = sum(precision_at_k_results[k]) / len(precision_at_k_results[k])
        else:
            mean_precision_at_k[k] = 0
        print(f"Mean Precision@{k}: {mean_precision_at_k[k]:.4f}")

    # Return both individual results, MAP, and mean precision@k
    return results, map_score, mean_precision_at_k

def display_evaluation_summary(eval_results, map_score=None, mean_precision_at_k=None):
    """
    Display summary of evaluation results with plots shown inline
    """
    # Calculate overall metrics
    avg_precision = sum(res['precision'] for res in eval_results.values()) / len(eval_results) if eval_results else 0
    avg_score = sum(res['avg_score'] for res in eval_results.values()) / len(eval_results) if eval_results else 0
    avg_ap = sum(res.get('average_precision', 0) for res in eval_results.values()) / len(eval_results) if eval_results else 0

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Number of test queries: {len(eval_results)}")
    print(f"Average category precision: {avg_precision:.2%}")
    print(f"Average relevance score: {avg_score:.4f}")
    print(f"Mean Average Precision (MAP): {map_score if map_score is not None else avg_ap:.4f}")

    # Display mean precision@k values
    if mean_precision_at_k:
        for k, value in sorted(mean_precision_at_k.items()):
            print(f"Mean Precision@{k}: {value:.4f}")

    # Create evaluation metrics table
    print("\nPer-Query Evaluation Metrics:")
    print(f"{'-'*90}")
    header = f"{'Query ID':<10}{'Query':<30}{'Precision':<12}{'Avg Precision':<15}{'Relevance':<12}"
    if mean_precision_at_k:
        for k in sorted(mean_precision_at_k.keys()):
            header += f"P@{k:<6}"
    print(header)
    print(f"{'-'*90}")

    for i, (query_id, res) in enumerate(eval_results.items()):
        query_text = res['query']
        if len(query_text) > 25:
            query_text = query_text[:22] + "..."

        row = f"{f'Q{i+1}':<10}{query_text:<30}{res['precision']:.2%}{res.get('average_precision', 0):.4f}{res['avg_score']:.4f}"
        if mean_precision_at_k:
            for k in sorted(mean_precision_at_k.keys()):
                p_at_k = res.get('precision_at_k', {}).get(k, 0)
                row += f"{p_at_k:.4f}  "
        print(row)

    # Create bar chart for precision and AP by query - SHOWN INLINE
    plt.figure(figsize=(12, 6))
    queries = [f"Q{i+1}" for i in range(len(eval_results))]
    precisions = [res['precision'] for res in eval_results.values()]
    aps = [res.get('average_precision', 0) for res in eval_results.values()]

    x = np.arange(len(queries))
    width = 0.35

    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, aps, width, label='Average Precision')

    plt.axhline(y=avg_precision, color='r', linestyle='--', label=f'Avg Precision: {avg_precision:.2%}')
    plt.axhline(y=avg_ap, color='g', linestyle='--', label=f'MAP: {avg_ap:.4f}')

    plt.ylim(0, 1.1)
    plt.title('Precision Metrics by Query')
    plt.xlabel('Query')
    plt.ylabel('Score')
    plt.xticks(x, queries)
    plt.legend()
    plt.tight_layout()
    plt.show()  # Show directly instead of saving

    # Create Precision@k plot - SHOWN INLINE
    if mean_precision_at_k:
        plt.figure(figsize=(10, 6))
        k_values = sorted(mean_precision_at_k.keys())

        # For each query, plot precision@k
        for i, (query_id, res) in enumerate(eval_results.items()):
            precision_values = [res.get('precision_at_k', {}).get(k, 0) for k in k_values]
            plt.plot(k_values, precision_values, marker='o', label=f'Q{i+1}')

        # Plot mean precision@k
        mean_values = [mean_precision_at_k[k] for k in k_values]
        plt.plot(k_values, mean_values, marker='s', linestyle='--', linewidth=2, color='black', label='Mean')

        plt.title('Precision@k for Test Queries')
        plt.xlabel('k')
        plt.ylabel('Precision@k')
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()  # Show directly instead of saving

def calculate_average_precision(results, target_category, documents):
    """
    Calculate average precision for a query

    Args:
        results: List of (doc_id, score) tuples
        target_category: Expected document category
        documents: Document collection

    Returns:
        float: Average precision
    """
    if not results:
        return 0.0

    # Track precision at each relevant document
    precisions = []
    relevant_count = 0

    for i, (doc_id, _) in enumerate(results):
        rank = i + 1
        # Check if document matches target category
        if documents[doc_id]['category'] == target_category:
            relevant_count += 1
            # Calculate precision at this rank
            precision_at_rank = relevant_count / rank
            precisions.append(precision_at_rank)

    # Return average of precisions at relevant documents
    if precisions:
        return sum(precisions) / len(precisions)
    else:
        return 0.0

def calculate_precision_recall_curve(search_results, target_category, documents):
    """
    Calculate precision-recall curve data points
    """
    if not search_results:
        return [0], [0]

    # Calculate precision and recall at each rank
    precisions = []
    recalls = []
    relevant_count = 0

    # Count total relevant documents (matching target category)
    total_relevant = sum(1 for doc_id, _ in search_results
                         if documents[doc_id]['category'] == target_category)

    if total_relevant == 0:
        return [0], [0]

    for i, (doc_id, _) in enumerate(search_results):
        if documents[doc_id]['category'] == target_category:
            relevant_count += 1

        # Calculate precision and recall at this rank
        precision = relevant_count / (i + 1)
        recall = relevant_count / total_relevant

        precisions.append(precision)
        recalls.append(recall)

    return recalls, precisions

def calculate_11point_interpolated_precision(recalls, precisions):
    """
    Calculate 11-point interpolated precision

    Args:
        recalls: List of recall values
        precisions: List of precision values

    Returns:
        dict: 11-point interpolated precision values
    """
    interpolated = {}

    # Standard recall levels
    std_recalls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Ensure recalls and precisions are paired correctly
    points = list(zip(recalls, precisions))
    points.sort(key=lambda x: x[0])  # Sort by recall

    # Add a point for recall=1.0 if not present
    if points and points[-1][0] < 1.0:
        points.append((1.0, 0.0))

    # Calculate interpolated precision for each standard recall level
    for std_recall in std_recalls:
        # Find precision values at recall >= std_recall
        precisions_at_recall = [p for r, p in points if r >= std_recall]

        if precisions_at_recall:
            # Interpolated precision is maximum precision at recall >= std_recall
            interpolated[std_recall] = max(precisions_at_recall)
        else:
            interpolated[std_recall] = 0.0

    return interpolated

def plot_precision_recall_curves(test_queries, documents, inverted_index, doc_vectors):
    """
    Plot precision-recall curves for all test queries - shown inline
    """
    plt.figure(figsize=(10, 6))

    # Process each query
    for i, query_info in enumerate(test_queries):
        query = query_info['query']
        target_category = query_info['category']

        # Get search results
        search_results = search(query, documents, inverted_index, doc_vectors)

        # Calculate precision-recall curve
        recalls, precisions = calculate_precision_recall_curve(search_results, target_category, documents)

        # Plot curve
        plt.plot(recalls, precisions, marker='.', label=f"Q{i + 1}: {query_info['description']}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Test Queries')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()  # Show directly instead of saving

def compare_personalized_search_results(query, user_profiles, documents, inverted_index, doc_vectors):
    """
    Compare search results for the same query across different user profiles
    """
    print(f"\n{'=' * 80}")
    print(f"SEARCH RESULTS COMPARISON FOR QUERY: '{query}'")
    print(f"{'=' * 80}")

    # First get non-personalized results as baseline
    print(f"\nBASELINE (Non-personalized) Results:")
    print(f"{'-' * 40}")
    base_results = search(query, documents, inverted_index, doc_vectors)
    display_search_results(base_results, documents, 5)

    # Get personalized results for each user
    for user_id in user_profiles:
        print(f"\n{user_id.upper()} Personalized Results:")
        print(f"{'-' * 40}")
        user_results = search(query, documents, inverted_index, doc_vectors, user_id, user_profiles)
        display_search_results(user_results, documents, 5)

        # Find differences in ranking
        baseline_docs = [doc_id for doc_id, _ in base_results[:5]]
        personalized_docs = [doc_id for doc_id, _ in user_results[:5]]

        # New documents that appear in personalized results
        new_docs = [doc_id for doc_id in personalized_docs if doc_id not in baseline_docs]

        if new_docs:
            print(f"\nDocuments promoted due to {user_id}'s profile:")
            for doc_id in new_docs:
                doc = documents[doc_id]
                print(f"- [{doc_id}] {doc['category']}/{doc['name']}")

                # Explain why it was promoted
                if doc['category'] in user_profiles[user_id]['top_categories']:
                    category_rank = user_profiles[user_id]['top_categories'].index(doc['category']) + 1
                    print(f"  (Category '{doc['category']}' is #{category_rank} in user interests)")

                matching_terms = []
                for term in user_profiles[user_id]['interest_vector']:
                    if term in doc['tokens']:
                        matching_terms.append(term)

                if matching_terms:
                    print(f"  (Content matches user interests: {', '.join(matching_terms[:5])})")

    return base_results

def main():
    """
    Main function to run the content-based recommender system
    """
    # Set the base directory for the dataset
    base_dir = 'bbc_articles'  # Update this to your dataset path

    # Step A: Load and preprocess documents
    print("\n" + "="*70)
    print("A. LOADING AND PREPROCESSING DOCUMENTS")
    print("="*70)
    documents = load_documents(base_dir)
    documents = preprocess_documents(documents)

    # Step B: Build inverted index and calculate TF-IDF
    print("\n" + "="*70)
    print("B. BUILDING INVERTED INDEX AND CALCULATING TF-IDF")
    print("="*70)
    documents = calculate_term_frequencies(documents)
    inverted_index = build_inverted_index(documents)
    tfidf, doc_vectors = calculate_tfidf(documents, inverted_index)

    # Step C: Document similarity demonstration
    print("\n" + "="*70)
    print("C. DOCUMENT SIMILARITY DEMONSTRATION")
    print("="*70)

    # Choose a sample document for similarity demonstration
    if documents:
        sample_doc_id = list(documents.keys())[0]
        sample_doc = documents[sample_doc_id]
        print(f"\nFinding similar documents to: [{sample_doc_id}] {sample_doc['category']}/{sample_doc['name']}")

        similar_docs = get_similar_documents(sample_doc_id, doc_vectors, documents)

    # Step D: Create user profiles based on the dataset
    print("\n" + "="*70)
    print("D. CREATING USER PROFILES AND DEMONSTRATING PERSONALIZED SEARCH")
    print("="*70)

    # Create user profiles based on the actual document categories
    user_profiles = create_user_profiles(documents)

    # Generate test queries based on document content
    test_queries = create_test_queries(documents)

    # Choose a sample query for personalization demonstration
    if test_queries:
        sample_query = test_queries[0]['query']

        # Compare personalized search results
        compare_personalized_search_results(sample_query, user_profiles, documents, inverted_index, doc_vectors)

    # Step E: Evaluate the recommender system
    print("\n" + "="*70)
    print("E. EVALUATING THE RECOMMENDER SYSTEM")
    print("="*70)

    # Evaluate search
    eval_results, map_score, mean_precision_at_k = evaluate_search(test_queries, documents, inverted_index, doc_vectors)

    # Display evaluation summary
    display_evaluation_summary(eval_results, map_score, mean_precision_at_k)

    # Plot precision-recall curves
    plot_precision_recall_curves(test_queries, documents, inverted_index, doc_vectors)

    # Create a MAP table
    print("\nMAP Evaluation Table:")
    print("=" * 60)
    print(f"{'Method':<30}{'MAP Score':<15}{'Notes':<25}")
    print("-" * 60)
    print(f"{'Base Recommender System':<30}{map_score:.4f}{'':<25}")

    # Compare with variants (e.g., with/without tolerance)
    # Variant 1: Without tolerant retrieval
    # Run a subset of queries for efficiency
    subset_queries = test_queries[:min(2, len(test_queries))]

    # Run evaluation with modified search parameters
    modified_results = []
    for query_info in subset_queries:
        query = query_info['query']
        # Use search without tolerance (exact matching only)
        results = search(query, documents, inverted_index, doc_vectors, tolerance=1.0)  # 1.0 means exact match only
        ap = calculate_average_precision(results, query_info['category'], documents)
        modified_results.append(ap)

    map_no_tolerance = sum(modified_results) / len(modified_results) if modified_results else 0
    print(f"{'Without Tolerant Retrieval':<30}{map_no_tolerance:.4f}{'Using exact matching only':<25}")

    # Variant 2: With personalization (for first user)
    if user_profiles:
        user_id = list(user_profiles.keys())[0]
        results_personalized = []
        for query_info in subset_queries:
            query = query_info['query']
            results = search(query, documents, inverted_index, doc_vectors, user_id, user_profiles)
            ap = calculate_average_precision(results, query_info['category'], documents)
            results_personalized.append(ap)

        map_personalized = sum(results_personalized) / len(results_personalized) if results_personalized else 0
        print(f"{'With Personalization (' + user_id + ')':<30}{map_personalized:.4f}{'Using user profile':<25}")

    print("=" * 60)

    # Display search histories from all user profiles
    print("\nUser Search Histories:")
    print("=" * 60)
    for user_id, profile in user_profiles.items():
        print(f"\n{user_id}:")
        for i, query in enumerate(profile['search_history']):
            print(f"  {i+1}. {query}")

    # Optional: Allow user to input a custom query
    print("\n" + "="*70)
    print("INTERACTIVE SEARCH MODE")
    print("="*70)
    user_query = input("Enter your search query (or press Enter to skip): ")

    # Even if no query is provided, show evaluation results from test queries
    if not user_query and test_queries:
        print("\nUsing first test query for demonstration since no query was provided")
        user_query = test_queries[0]['query']

    if user_query:  # Will be true from either user input or test query fallback
        print("\nNon-personalized search results:")
        results = search(user_query, documents, inverted_index, doc_vectors)
        display_search_results(results, documents, 10)

        # If user profiles exist, show personalized results for all profiles
        if user_profiles:
            for user_id in user_profiles:
                print(f"\nPersonalized search results for {user_id}:")
                print(f"{'-' * 40}")
                personalized_results = search(user_query, documents, inverted_index, doc_vectors,
                                              user_id, user_profiles)
                display_search_results(personalized_results, documents, 10)

if __name__ == "__main__":
    main()
