# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is downloaded (run this once)
for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# --- Configuration ---
URLS = [
    "https://www.strategicmarketresearch.com/market-report/power-drill-market",
    "https://www.marketdataforecast.com/market-reports/north-america-power-tools-market" ,
    "https://www.researchandmarkets.com/report/power-drill",
    "https://www.grandviewresearch.com/industry-analysis/power-tools-market",
    "https://www.gminsights.com/industry-analysis/electric-power-tools-market",
    "https://www.maximizemarketresearch.com/market-report/global-power-drill-market/88127/",
    "https://www.grandviewresearch.com/industry-analysis/rotary-hammer-drill-market",
    "https://www.grandviewresearch.com/industry-analysis/us-power-tools-market-report"
]
OUTPUT_DIR = "word_analysis_results"
TOP_N_WORDS = 20
TOP_N_KEYWORDS = 30  # Number of topic keywords to keep per article for filtering

def fetch_text_from_url(url):
    """Fetches and extracts visible text content from a given URL."""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=' ')
        return ' '.join(text.split())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def preprocess_text(text):
    """Cleans and preprocesses English text: lowercase, tokenize, remove punctuation and stopwords."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

def extract_keywords_tfidf(docs, top_n=TOP_N_KEYWORDS):
    """
    Extracts topic-relevant keywords from each document using TF-IDF.
    Returns a list of sets, each containing the top keywords for that doc.
    """
    # Use TfidfVectorizer to get important words
    tfidf = TfidfVectorizer(
        stop_words='english',
        tokenizer=word_tokenize,
        token_pattern=None,  # necessary since we override tokenizer
        lowercase=True,
        max_df=0.85,  # Ignore very common terms
        max_features=1000
    )
    tfidf_matrix = tfidf.fit_transform(docs)
    feature_names = tfidf.get_feature_names_out()
    keywords_per_doc = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(doc_idx)
        scores = list(zip(row.indices, row.data))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_keywords = {feature_names[idx] for idx, _ in scores[:top_n]}
        keywords_per_doc.append(top_keywords)
    return keywords_per_doc

def generate_word_cloud(word_counts, filename):
    """Generates and saves a word cloud from word counts."""
    if not word_counts:
        print(f"No words to generate word cloud for {filename}")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    print(f"Word cloud saved to {filename}")

def generate_heatmap(word_counts, filename, top_n=20):
    """Generates and saves a heatmap of top N word frequencies."""
    top_words = word_counts.most_common(top_n)
    if not top_words:
        print("No words found to generate heatmap.")
        return
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    df = df.set_index('Word')
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="d", cmap="viridis")
    plt.title(f'Top {top_n} Word Frequency Heatmap')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved to {filename}")

def generate_frequency_table(word_counts, filename, top_n=20):
    """Generates and saves a frequency table (CSV) of top N words."""
    top_words = word_counts.most_common(top_n)
    if not top_words:
        print("No words found to generate frequency table.")
        return
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    df.to_csv(filename, index=False)
    print(f"Frequency table saved to {filename}")

def analyze_articles(urls):
    """Main pipeline to fetch, process, and analyze articles."""
    docs = []
    url_text_map = {}
    for url in urls:
        print(f"Fetching and processing {url}...")
        text = fetch_text_from_url(url)
        if text:
            docs.append(text)
            url_text_map[url] = text
        else:
            print(f"Skipping {url} due to fetch error.")
    if not docs:
        print("No articles could be fetched or processed.")
        return

    # Extract topic-relevant keywords for each article
    keywords_per_doc = extract_keywords_tfidf(docs, top_n=TOP_N_KEYWORDS)

    all_tokens = []
    filtered_tokens = []
    for idx, (url, text) in enumerate(url_text_map.items()):
        tokens = preprocess_text(text)
        all_tokens.extend(tokens)
        keywords = keywords_per_doc[idx]
        filtered = [tok for tok in tokens if tok in keywords]
        filtered_tokens.extend(filtered)

    print(f"Total words processed (after cleaning): {len(all_tokens)}")
    print(f"Total topic-relevant words processed: {len(filtered_tokens)}")

    all_word_counts = Counter(all_tokens)
    filtered_word_counts = Counter(filtered_tokens)
    print(f"Unique words found: {len(all_word_counts)}")
    print(f"Unique topic-relevant words: {len(filtered_word_counts)}")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Generate outputs for all words (optional)
    # generate_word_cloud(all_word_counts, os.path.join(OUTPUT_DIR, "word_cloud_all.png"))

    # Generate outputs for filtered topic-relevant words
    wordcloud_file = os.path.join(OUTPUT_DIR, "word_cloud_topic.png")
    heatmap_file = os.path.join(OUTPUT_DIR, "frequency_heatmap_topic.png")
    table_file = os.path.join(OUTPUT_DIR, "frequency_table_topic.csv")

    generate_word_cloud(filtered_word_counts, wordcloud_file)
    generate_heatmap(filtered_word_counts, heatmap_file, top_n=TOP_N_WORDS)
    generate_frequency_table(filtered_word_counts, table_file, top_n=TOP_N_WORDS)

    print("\nAnalysis complete. Results saved in:")
    print(f"- Word Cloud (topic-relevant): {wordcloud_file}")
    print(f"- Heatmap (topic-relevant): {heatmap_file}")
    print(f"- Frequency Table (topic-relevant): {table_file}")
    print(f"\nTop {TOP_N_WORDS} topic-relevant words:")
    for word, count in filtered_word_counts.most_common(TOP_N_WORDS):
        print(f"- {word}: {count}")

if __name__ == "__main__":
    if not URLS or all(url.startswith("https://example.com") for url in URLS):
        print("Error: Please update the 'URLS' list in the script with the actual web page URLs you want to analyze.")
    else:
        analyze_articles(URLS)
