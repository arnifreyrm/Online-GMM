import numpy as np


def gaussian_data_with_step_stream(means, covs, steps, points_per_cluster):
    k = len(means)

    local_means = [m.astype(float) for m in means]
    local_steps = [s.astype(float) for s in steps]

    for _ in range(points_per_cluster):
        for cluster_idx in range(k):
            point = np.random.multivariate_normal(
                local_means[cluster_idx], covs[cluster_idx])
            yield point
        for i in range(k):
            local_means[i] += local_steps[i]


velocity = 0.01
means = [
    np.array([-15, 0], dtype=float),
    np.array([10, 10], dtype=float),
    np.array([-15, 15], dtype=float),
    np.array([5, -7], dtype=float),
    np.array([30, 30], dtype=float),
    np.array([0, 0], dtype=float)
]
steps = [
    np.array([-velocity, 0], dtype=float),
    np.array([velocity, 0], dtype=float),
    np.array([velocity, velocity], dtype=float),
    np.array([velocity, -velocity], dtype=float),
    np.array([velocity, 0], dtype=float),
    np.array([0, velocity], dtype=float)
]
covs = [
    np.eye(2) * 0.5,
    np.eye(2) * 0.8,
    np.eye(2) * 0.6,
    np.eye(2) * 0.2,
    np.eye(2) * 0.9,
    np.eye(2) * 0.3
]

points_per_cluster = 1000
stream = gaussian_data_with_step_stream(means, covs, steps, points_per_cluster)

points = []
for p in stream:
    points.append(p)

np_points = np.array(points)
# np.save("./Online-GMM/gaussian_drift_points.npy", np_points)


def gaussian_drift_true_means(means, covs, steps, points_per_cluster):
    k = len(means)

    local_means = [m.astype(float) for m in means]
    local_steps = [s.astype(float) for s in steps]
    true_mean_history = []
    for _ in range(points_per_cluster):
        for cluster_idx in range(k):
            true_mean_history.append([m.astype(float) for m in local_means])
        for i in range(k):
            local_means[i] += local_steps[i]

    return np.array(true_mean_history)


# Parameters
velocity = 0.01
means = [
    np.array([-15, 0], dtype=float),
    np.array([10, 10], dtype=float),
    np.array([-15, 15], dtype=float),
    np.array([5, -7], dtype=float),
    np.array([30, 30], dtype=float),
    np.array([0, 0], dtype=float)
]
steps = [
    np.array([-velocity, 0], dtype=float),
    np.array([velocity, 0], dtype=float),
    np.array([velocity, velocity], dtype=float),
    np.array([velocity, -velocity], dtype=float),
    np.array([velocity, 0], dtype=float),
    np.array([0, velocity], dtype=float)
]
covs = [
    np.eye(2) * 0.5,
    np.eye(2) * 0.8,
    np.eye(2) * 0.6,
    np.eye(2) * 0.2,
    np.eye(2) * 0.9,
    np.eye(2) * 0.3
]
points_per_cluster = 1000

# Generate drifting means
true_mean_history = gaussian_data_with_step_stream(
    means, covs, steps, points_per_cluster)

# Print the shape of the resulting history
print(true_mean_history.shape)  # Expected output: (6000 * 6, 6, 2)

# Save the array if needed
np.save("./Online-GMM/gaussian_drift_true_centers.npy", true_mean_history)
