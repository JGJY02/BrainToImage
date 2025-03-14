
import matplotlib.pyplot as plt
from keras import Model
from sklearn.manifold import TSNE


def extract_features(model, data):
    # Create a feature extraction model
    feature_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Assuming second to last layer
    features = feature_layer_model.predict(data)
    return features

def apply_tsne(features, perplexity=30, n_components=2, random_state=0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

# Plotting function
def plot_tsne(tsne_results, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE Visualization of Class Separability')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()