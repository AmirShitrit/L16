import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


def reduce_dimensions(sentence_vectors):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(sentence_vectors.toarray())


def plot_clusters(coordinates, labels, num_clusters):
    colors = ['blue', 'red', 'green']

    plt.figure(figsize=(12, 8))

    for cluster_id in range(num_clusters):
        cluster_points = coordinates[labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[cluster_id],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=100
        )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-Means Clustering of Sentences (K=3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('clusters_plot.png', dpi=300)
    print("\nScatter plot saved as 'clusters_plot.png'")
    plt.show()


def main():
    sentences = load_sentences('sentences.json')
    sentence_vectors = vectorize_sentences(sentences)
    labels = cluster_sentences(sentence_vectors, num_clusters=3)
    clusters = group_by_cluster(sentences, labels, num_clusters=3)
    display_clusters(clusters)

    coordinates = reduce_dimensions(sentence_vectors)
    plot_clusters(coordinates, labels, num_clusters=3)


if __name__ == '__main__':
    main()
