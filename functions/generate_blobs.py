from gaussian_stream import gaussian_data_stream
import numpy as np

stream = gaussian_data_stream(300)

points = []
for point in range(500*6):
    points.append(next(stream))
np_points = np.array(points)
# save to a npy file
np.save("./Online-GMM/dataset/6_Gauss_Blobs.npy", np_points)
