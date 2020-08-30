from functools import reduce
import pathlib
import json

import torch
import torch.nn as nn


class ModelSaverLoader:
    """
    Saves and loads models to/from disk.
    """

    CONFIG_FILE_NAME = "config.json"
    PARAMS_FILE_NAME = "params"

    @classmethod
    def save(cls, model, folder_path):
        folder_path = pathlib.Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        model_config = {
            "map_shape": model._map_shape,
            "node_dims": model._node_dims
        }
        with open(folder_path.joinpath(cls.CONFIG_FILE_NAME), 'w', encoding="utf-8") as fp:
            json.dump(model_config, fp, indent=2)

        torch.save(model.state_dict(), folder_path.joinpath(cls.PARAMS_FILE_NAME))

    @classmethod
    def load(cls, folder_path, dtype=torch.float32, device=None):
        folder_path = pathlib.Path(folder_path)
        with open(folder_path.joinpath(cls.CONFIG_FILE_NAME), 'r', encoding="utf-8") as fp:
            model_config = json.load(fp)

        model = SOM(tuple(model_config["map_shape"]), model_config["node_dims"], dtype=dtype, device=device)
        model.load_state_dict(torch.load(folder_path.joinpath(cls.PARAMS_FILE_NAME)))

        return model


class SOM(nn.Module):
    """
    Self-Organizing Map using Gaussian-like neighbourhood function and dot-product similarity.
    The model caches the squared distances between all map nodes.  This requires N^2 x size(dtype) of memory.
    If you are training on GPU and you do not have enough memory for this you will have to modify this code.
    """

    def __init__(self, map_shape, node_dims, dtype=torch.float32, device=None):
        """
        :param map_shape: A tuple of dimensions.  e.g. (20, 20, 20)
        :param node_dims: A scalar specifying number of dimensions used for each node.
        """
        super().__init__()
        self._map_shape = map_shape
        self._node_dims = node_dims
        self._dtype = dtype
        self._device = device

        self._num_nodes = reduce(lambda x, y: x * y, self._map_shape)
        self._weights = nn.parameter.Parameter(torch.randn((self._num_nodes, node_dims), dtype=dtype, device=device), requires_grad=False)
        self._normalize_weights()
        self._node_indices = nn.parameter.Parameter(self._create_node_indices().to(dtype=dtype, device=device), requires_grad=False)
        self._map_space_distances_squared = nn.parameter.Parameter(self._compute_map_space_distances_squared().to(device=device, dtype=dtype), requires_grad=False)

    @property
    def weights(self):
        return self._weights.view(self._map_shape + (self._node_dims,))

    @property
    def shape(self):
        return self._map_shape

    def forward(self, x, alpha=0.0, sigma_squared=1.0):
        """
        Activates nodes in response to x where x is a single input vector.
        Returns the softmaxed activation of all the nodes in the map during inference: raw activations during training.
        """
        prod = torch.matmul(self._weights, x)
        if self.training:
            bmu_index = torch.argmax(prod)
            influence = self._compute_influence(bmu_index, alpha, sigma_squared)
            self._update_weights(x, influence)
        return nn.functional.softmax(prod.flatten(), dim=0).view(self._map_shape)

    def bmu(self, x):
        """
        Returns the N dimensional index of the best matching unit for input vector x, where N is the number of map dimensions.
        """
        return self._node_indices[torch.argmax(torch.matmul(self._weights, x))]

    def _compute_map_space_distances_squared(self):
        """
        For each node in the map, computes the squared L2 distance (in map space) to each other node in the map.
        Returns an M x M matrix where M is the number of nodes in the map.
        """
        m = self._num_nodes

        return torch.sum(
            torch.pow(
                torch.repeat_interleave(self._node_indices, m, dim=0) - self._node_indices.repeat((m, 1)),
                2.0),
            1).view(m, m)

    def _create_node_indices(self):
        """
        Creates an M x N tensor of indices where M is the number of map nodes and N is the number of map dimensions.
        """
        shape = self._map_shape
        n = len(shape)
        m = self._num_nodes
        node_indices = torch.zeros(shape + (n,)).view(m, n)

        num_per_dim = m
        for d in range(n):
            num_per_dim = num_per_dim / shape[d]
            for i in range(m):
                node_indices[i][d] = int(i / num_per_dim) % shape[d]
        return node_indices

    def _compute_influence(self, bmu_index, alpha, sigma_squared):
        """
        Computes the update influence over the whole map given alpha, sigma squared, and the bmu.
        Returns a 1 dimensional tensor.
        """
        return torch.mul(torch.exp(torch.neg(torch.div(self._map_space_distances_squared[bmu_index], sigma_squared))), alpha)

    def _update_weights(self, x, influence):
        self._weights = nn.parameter.Parameter(self._weights + torch.mul(influence.view((self._num_nodes, 1)), x.repeat((self._num_nodes, 1))), requires_grad=False)
        self._normalize_weights()

    def _normalize_weights(self):
        self._weights = nn.parameter.Parameter(torch.div(self._weights, torch.norm(self._weights, dim=1).view((self._num_nodes, 1))), requires_grad=False)
