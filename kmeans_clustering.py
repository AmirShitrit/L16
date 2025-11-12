import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def load_sentences(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def vectorize_sentences(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences)


def cluster_sentences(sentence_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans.fit_predict(sentence_vectors)


def group_by_cluster(sentences, labels, num_clusters):
    clusters = {i: [] for i in range(num_clusters)}
    for sentence, label in zip(sentences, labels):
        clusters[label].append(sentence)
    return clusters


def display_clusters(clusters):
    for cluster_id, sentences in clusters.items():
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id} ({len(sentences)} sentences)")
        print('='*80)
        for sentence in sentences:
            print(f"  - {sentence}")


def main():
    sentences = load_sentences('sentences.json')
    sentence_vectors = vectorize_sentences(sentences)
    labels = cluster_sentences(sentence_vectors, num_clusters=3)
    clusters = group_by_cluster(sentences, labels, num_clusters=3)
    display_clusters(clusters)


if __name__ == '__main__':
    main()
