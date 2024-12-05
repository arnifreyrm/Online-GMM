import numpy as np

means = [np.array([-15, 0]), np.array([10, 10]), np.array([-15, 15]),
         np.array([5, -7]), np.array([30, 30]), np.array([0, 0])]
covs = [np.eye(2) * 0.5, np.eye(2) * 0.8, np.eye(2) * 0.6,
        np.eye(2)*0.2, np.eye(2)*0.9, np.eye(2)*0.3]


def gaussian_data_stream(points_per_cluster):
    """
    Generate a stream of data points sampled from k 2D Gaussian distributions.

    Parameters:
        means (list of np.ndarray): List of mean vectors for each Gaussian (2D).
        covs (list of np.ndarray): List of covariance matrices for each Gaussian (2D).
        points_per_cluster (int): Number of points per cluster.

    Yields:
        np.ndarray: A random 2D data point sampled from one of the Gaussians.
    """
    k = len(means)
    cluster_sizes = [points_per_cluster] * k
    while True:
        # Randomly choose a Gaussian distribution
        cluster = np.random.choice(k)
        # Sample a point from the chosen Gaussian
        yield np.random.multivariate_normal(means[cluster], covs[cluster])
