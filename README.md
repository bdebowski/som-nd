# som-nd
PyTorch implementation of Self Organizing Map allowing N-Dimensional maps (where N can be any positive integer).  Uses dot-product similarity (implemented as proposed in [1]) instead of Euclidean distance.  I wrote this because existing pytorch som implementations I found were limited to 2 dimensional maps and did not provide usage for dot-product similarity.  I also needed some practice working with pytorch.

## Usage
See the example.py file for usage example.  Training code is provided to simplify usage.  The training code will automatically save checkpoints from which it can be restored if training is interrupted.

## Caveats
1. The training code included assumes that your entire training dataset resides in a Tensor on the same device as your model.  If you are training on GPU and have a large dataset this means you may not have enough GPU memory to use the training code provided.  You can write your own training loop following the loop provided as an example.

2. The model computes the distances between all nodes in the map and caches these values to speed up training.  This caching requres M^2 x sizeof(dtype) of device memory, where M is the number of nodes in the map.  If you are creating a large map and training on GPU this means you may not have enough GPU memory available to run this code.

## Requirements
- pytorch >= 1.5.0
- matplotlib = 3.3.1 (just for the example code)

## References
[1] Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (1990): 1464-1480.
