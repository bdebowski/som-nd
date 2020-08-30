"""
Example to illustrate usage.  We build a 3-dimensional map to show that it is possible to do so.
Note that the dot-product similarity used is likely not ideal for this use case of organizing colours (Euclidean distance would probably be better).
Colours are used because they illustrate the mechanics of the SOM in an easy to understand way.
"""

import torch
import matplotlib.pyplot as plt

from src.som.model import SOM, ModelSaverLoader
from src.som.training import ModelTraining


DEVICE = "cuda:0"
DTYPE = torch.float32
X, Y, Z = 12, 12, 12


# Training set will be 1000 random RGB colours
colours = torch.rand((1000, 3), device=DEVICE, dtype=DTYPE)

# Iteration setup
# The problem is trivial so we divide the recommended number of steps by 10 to speed up training
n_total_steps = int(ModelTraining.recommended_num_steps((X, Y, Z)) / 10)

# Build model
som = SOM((X, Y, Z), 3, device=DEVICE, dtype=DTYPE)

# Train
# We set the save_interval to 1 minute for demonstration purposes
ModelTraining.run(colours, n_total_steps, som, r"C:\temp\out", save_interval_min=1, dtype=DTYPE, device=DEVICE)

# Load saved model and switch to inference mode
som = ModelSaverLoader.load(r"C:\temp\out\saved-model", dtype=DTYPE, device=DEVICE)
som.eval()

# Plot Colours
# We plot each of the slices of the 3D map
weights = som.weights.to(device="cpu").numpy()
fig, axs = plt.subplots(3, 4)
for i in range(12):
    axs.flat[i].imshow(weights[i], interpolation='gaussian')
    axs.flat[i].axis('off')
fig.show()
