# 🚀 Machine Learning Project Generation Prompt

**Objective:** Develop a complete, highly structured, and executable software project that applies unsupervised clustering (K-Means) to text data, followed by supervised classification (KNN) of new data, complete with visualization, automation, and professional documentation.

---

## 💻 Part 1: Core Code Generation (`classifier.py`)

Generate a single, robust **Python program (`classifier.py`)** that executes the following machine learning and data processing tasks:

1.  **Data Ingestion:** Read and load sentences from two separate JSON files: `@sentences.json` (training data) and `@new_sentences.json` (test data).
2.  **Vectorization:** Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to numerically vectorize all sentences. The vectorizer must be fitted only on the training data (`@sentences.json`).
3.  **Clustering:** Apply the **K-Means algorithm** with a fixed $K=3$ to the vectorized training data.
4.  **Output Generation:** Create a new JSON file named **`sentences_clustered.json`**. This file must contain the original sentences organized into three distinct lists, one for each cluster found by K-Means (e.g., `Cluster 0`, `Cluster 1`, `Cluster 2`).
5.  **Classification:**
    * Train a **K-Nearest Neighbors (KNN)** classifier with a fixed $K=5$, using the K-Means cluster assignments as the class labels for the training data.
    * Vectorize the sentences from `@new_sentences.json` using the pre-fitted TF-IDF model.
    * Use the trained KNN model to predict the cluster assignment for each new sentence.
6.  **Console Output:** Print a list to the console showing each new sentence and the predicted cluster (0, 1, or 2) it was attached to by the KNN model.

---

## 📊 Part 2: Visualization and Automation

1.  **Visualization (`results.png`):**
    * Implement data visualization using **Matplotlib** to generate a single image file named **`results.png`**.
    * **Dimensionality Reduction:** Use **PCA (Principal Component Analysis)** to reduce the high-dimensional feature vectors to 2 dimensions for plotting.
    * **K-Means Plot:** Plot the 2D reduced training data, using three distinct colors to represent the three K-Means clusters.
    * **KNN Plot:** Plot the 2D reduced new sentences on the *same* chart. Use the color corresponding to their assigned cluster, but use a highly visible, contrasting **circle/ring marker** (e.g., a thick black border) to visually distinguish them from the K-Means training points