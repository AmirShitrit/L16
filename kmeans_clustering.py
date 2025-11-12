import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_sentences(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def vectorize_sentences(sentences):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    return vectorizer, vectors


def cluster_sentences(sentence_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(sentence_vectors)
    return kmeans, labels


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


def classify_new_sentences(sentence_vectors, labels, vectorizer, new_filepath, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(sentence_vectors, labels)

    new_sentences = load_sentences(new_filepath)
    new_vectors = vectorizer.transform(new_sentences)
    new_labels = knn.predict(new_vectors)
    return new_sentences, new_vectors, new_labels


def print_new_sentence_assignments(new_sentences, labels):
    print(f"\n{'='*80}")
    print(f"NEW SENTENCE CLUSTER ASSIGNMENTS ({len(new_sentences)} sentences)")
    print('='*80)
    for sentence, label in zip(new_sentences, labels):
        print(f"Cluster {label}: {sentence}")


def reduce_dimensions(sentence_vectors):
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(sentence_vectors.toarray())
    return pca, coordinates


def plot_clusters(coordinates, labels, num_clusters, new_coordinates=None, new_labels=None):
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

    if new_coordinates is not None and new_labels is not None:
        for cluster_id in range(num_clusters):
            new_cluster_points = new_coordinates[new_labels == cluster_id]
            if len(new_cluster_points) > 0:
                plt.scatter(
                    new_cluster_points[:, 0],
                    new_cluster_points[:, 1],
                    facecolors='none',
                    edgecolors=colors[cluster_id],
                    label=f'New (Cluster {cluster_id})',
                    linewidths=2,
                    s=150
                )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-Means Clustering (K=3) with KNN Classification (K=5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('clusters_plot.png', dpi=300)
    print("\nScatter plot saved as 'clusters_plot.png'")
    plt.show()


def main():
    num_clusters = 3

    sentences = load_sentences('sentences.json')
    vectorizer, sentence_vectors = vectorize_sentences(sentences)
    kmeans, labels = cluster_sentences(sentence_vectors, num_clusters)
    clusters = group_by_cluster(sentences, labels, num_clusters)
    display_clusters(clusters)

    pca, coordinates = reduce_dimensions(sentence_vectors)

    new_sentences, new_vectors, new_labels = classify_new_sentences(
        sentence_vectors, labels, vectorizer, 'new_sentences.json', k=5
    )
    print_new_sentence_assignments(new_sentences, new_labels)

    new_coordinates = pca.transform(new_vectors.toarray())

    plot_clusters(coordinates, labels, num_clusters, new_coordinates, new_labels)


if __name__ == '__main__':
    main()
