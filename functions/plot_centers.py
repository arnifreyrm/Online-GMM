import matplotlib.pyplot as plt
import numpy as np
import tqdm


def plot_gmm_trajectory_multiple(X, histories_mu, model_labels, title="Online Clustering for Drifting Gaussians", interval=1):
    """
    Plots the trajectories of GMM cluster centers over iterations for multiple models using scatter plots.

    Parameters:
    - X: The underlying datapoints in 2d, ndarray of shape (n_samples, 2)
        The dataset points.
    - histories_mu: list of ndarrays
        List of centers where each entry corresponds to a different model.
    - model_labels: list of str
        Labels for the models corresponding to the histories.
    - title: str, optional
        Title for the plot (default: "GMM Trajectory").
    - interval: int, optional
        Interval for plotting trajectory points (default: 1).
    """
    plt.figure(figsize=(10, 8))

    # Scatter underlying data points
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=1,
                alpha=0.1, label='Drifting Gaussian Blobs')

    num_models = len(histories_mu)
    colors = ['orange', 'green', 'blue']

    # Plot history and final cluster centers for each model
    for idx in range(num_models):
        history_mu = np.array(histories_mu[idx])
        model_label = model_labels[idx]
        color = colors[idx % len(colors)]
        T, n_components, _ = history_mu.shape
        s = 75
        marker = 'x'
        alpha = 1
        print(
            f"Plotting: {model_label}, Iterations: {T}, Clusters: {n_components}")

        # Scatter plot for trajectory points
        for t in tqdm.tqdm(range(0, T, interval), desc=f"Plotting {model_label}"):
            plt.scatter(
                history_mu[t, :, 0],
                history_mu[t, :, 1],
                color=color,
                s=5,
                alpha=0.01,
            )

        final_mu = history_mu[-1]
        plt.scatter(
            final_mu[:, 0],
            final_mu[:, 1],
            color=color,
            s=s,
            marker=marker,
            alpha=alpha,
            label=f"{model_label}"
        )

    # Plot labels and title
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
