import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st


class UnsupervisedLearningPipeline:
    """
    UnsupervisedLearningPipeline: A class for training, selecting, and evaluating unsupervised learning models, specifically for clustering tasks like K-Means.

    Attributes:
        pipeline (Pipeline or None): The pipeline used for preprocessing and model training. Initially None.
        best_model (Pipeline or None): The best-trained model pipeline. Initially None.
        results (dict): Stores results of training, with model names as keys and dictionaries containing pipeline and score.

    Methods:
        __init__(): Initializes the pipeline attributes.
        train(x, numeric_features, categorical_features, manual_k=None): Trains K-Means models with varying cluster counts (k), determines the optimal k, and trains the final model.
        train_with_custom_k(x, numeric_features, categorical_features, custom_k): Trains K-Means with a user-specified k.
        _train_kmeans(x, numeric_features, categorical_features, k): Builds and trains the K-Means pipeline with a specified number of clusters (k).
        _build_pipeline(model, numeric_features, categorical_features): Constructs a preprocessing and model pipeline.
        _plot_elbow_method(X, K, distortions): Plots the elbow method and silhouette scores to visualize and determine the optimal number of clusters.
    """


    def __init__(self):
        self.pipeline = None
        self.best_model = None
        self.results = {}

    def train(self, x, numeric_features, categorical_features,manual_k=None):
        self.x_train = x.copy()
        distortions = []
        K = range(1, 11)

        # Calculate distortions for different values of K
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            pipeline = self._build_pipeline(kmeans, numeric_features, categorical_features)
            pipeline.fit(x)
            distortions.append(pipeline.named_steps["model"].inertia_)

        # Call the plot method with the necessary arguments
        self._plot_elbow_method(x, K, distortions)

        # Find optimal K based on distortions
        optimal_k_elbow = distortions.index(min(distortions[1:])) + 1  # Skip the first value as it's not meaningful

        # Calculate silhouette scores
        silhouette_scores = []
        for k in K:
            if k > 1:  # Ensure there are at least two clusters
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(x)
                score = silhouette_score(x, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(None)  # No silhouette score for k=1

        # Determine the optimal k based on silhouette score (>= 0.5 threshold)
        threshold = 0.5
        optimal_k_silhouette = next((k for k, score in zip(K[1:], silhouette_scores[1:]) if score >= threshold), None)

        if manual_k is not None:
            optimal_k = manual_k
        elif optimal_k_silhouette is not None:
            optimal_k = min(optimal_k_elbow, optimal_k_silhouette)
        else:
            optimal_k = optimal_k_elbow

        self._train_kmeans(x, numeric_features, categorical_features, optimal_k)

    def train_with_custom_k(self, x, numeric_features, categorical_features, custom_k):
        self._train_kmeans(x, numeric_features, categorical_features, custom_k)

    def _train_kmeans(self, x, numeric_features, categorical_features, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        pipeline = self._build_pipeline(kmeans, numeric_features, categorical_features)
        pipeline.fit(x)
        self.best_model = pipeline
        self.results["KMeans"] = {"pipeline": pipeline, "score": -pipeline.named_steps["model"].inertia_}
        print(f"Model trained with k={k}")

    def _build_pipeline(self, model, numeric_features, categorical_features):
        return Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])),
            ("model", model)
        ])

    def _plot_elbow_method(self, X, K, distortions):
        silhouette_scores = []
        threshold = 0.5  # Define the silhouette score threshold

        # Calculate silhouette score for each K (skip k=1)
        for k in K:
            if k > 1:  # Ensure there are at least two clusters
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(None)  # No silhouette score for k=1

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot the distortions (Elbow method)
        ax1 = axes[0]
        ax1.plot(K, distortions, 'bx-', label='Distortion (Elbow)')
        optimal_k_elbow = distortions.index(min(distortions[1:])) + 1  # Skip first value as it's not meaningful
        ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--', label=f'Optimal k (Elbow) = {optimal_k_elbow}')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Distortion')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        ax1.legend()
        
        # Plot the silhouette scores for each k
        ax2 = axes[1]
        ax2.plot(K[1:], silhouette_scores[1:], 'go-', label='Silhouette Score')  # Skip k=1
        ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Silhouette Threshold = {threshold}')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for Optimal k')
        
        # Determine the optimal K based on silhouette score
        optimal_k_silhouette = next((k for k, score in zip(K[1:], silhouette_scores[1:]) if score >= threshold), None)
        if optimal_k_silhouette:
            ax2.axvline(x=optimal_k_silhouette, color='b', linestyle='--', label=f'Optimal k (Silhouette) = {optimal_k_silhouette}')
        
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
