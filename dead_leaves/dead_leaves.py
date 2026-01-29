import warnings
from pathlib import Path
from typing import Literal, Callable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import torch
import numpy as np
import pandas as pd
import PIL.Image
from torchvision.transforms.functional import pil_to_tensor

from .distributions import PowerLaw, Constant, Cosine, ExpCosine, Image
from .leaf_masks import circular, rectangular, ellipsoid, regular_polygon
from .utils import choose_compute_backend, bounding_box

warnings.filterwarnings("ignore", category=UserWarning)

dist_kw: dict[str, type[torch.distributions.distribution.Distribution]] = {
    "beta": torch.distributions.beta.Beta,
    "uniform": torch.distributions.uniform.Uniform,
    "normal": torch.distributions.normal.Normal,
    "poisson": torch.distributions.poisson.Poisson,
    "powerlaw": PowerLaw,
    "constant": Constant,
    "cosine": Cosine,
    "expcosine": ExpCosine,
    "image": Image,
}
"""Dictionary connecting keys to distribution classes."""

leaf_mask_kw: dict[
    str,
    Callable[
        [tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor] | pd.Series],
        torch.Tensor,
    ],
] = {
    "circular": circular,
    "ellipsoid": ellipsoid,
    "rectangular": rectangular,
    "polygon": regular_polygon,
}
"""Dictionary connecting keys to respective leaf mask generative function."""

dist_params: dict[str, set[str]] = {
    "beta": {"concentration0", "concentration1"},
    "uniform": {"low", "high"},
    "normal": {"loc", "scale"},
    "poisson": {"rate"},
    "powerlaw": {"low", "high", "k"},
    "constant": {"value"},
    "cosine": {"amplitude", "frequency"},
    "expcosine": {"frequency", "exponential_constant"},
    "image": {"dir"},
}

color_spaces = {
    ("B", "G", "R"): ("R", "G", "B"),
    ("H", "S", "V"): ("H", "S", "V"),
    ("gray",): ("gray"),
    ("source",): ("source"),
    ("alpha", "source"): ("source"),
}


class LeafGeometryGenerator:
    """
    Set up the geometrical dead leaves model.

    Args:
        leaf_shape (Literal["circular", "ellipsoid", "rectangular", "polygon"]):
            Shape of leaves.
        shape_param_distributions (dict[str, dict[str, [dict[str,float]]]):
            Leaf shape parameters and their distributions and distribution parameter values.
        image_shape (tuple[int, int]):
            Height (y, M) and width (x, N) of the area to be partitioned.
        position_mask (torch.Tensor, optional):
            Boolean tensor containing allowed leaf positions to create images with
            different shapes.
        n_sample (int, optional):
            Number of leaves to sample. If None, the sampling will stop when the full
            area is partitioned.
            Default is None.
        device (Literal["cuda", "mps", "cpu"]):
            Torch device to use, either 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        leaf_shape: Literal["circular", "ellipsoid", "rectangular", "polygon"],
        shape_param_distributions: dict[str, dict[str, dict[str, float]]],
        image_shape: tuple[int, int],
        position_mask: torch.Tensor | None = None,
        n_sample: int | None = None,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ) -> None:
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )
        """Chosen compute backend."""
        self.image_shape: tuple[int, int] = image_shape
        """Height (y, M) and width (x, N) of the canvas."""
        self.n_sample: int | None = n_sample
        """Number of leaves to sample."""
        self.leaf_shape: (
            Literal["circular"]
            | Literal["ellipsoid"]
            | Literal["rectangular"]
            | Literal["polygon"]
        ) = leaf_shape
        """Shape of the leaves."""
        self.shape_param_distributions: dict[str, dict[str, dict[str, float]]] = (
            shape_param_distributions
        )
        """Shape parameters and their distributions and distribution parameter values."""
        self.X, self.Y = torch.meshgrid(
            torch.arange(self.image_shape[1], device=self.device),
            torch.arange(self.image_shape[0], device=self.device),
            indexing="xy",
        )
        self.generate_leaf_mask: Callable = leaf_mask_kw[leaf_shape]
        """Method to generate mask of leaf on canvas."""
        self.position_mask = torch.ones(self.image_shape, dtype=int, device=self.device)
        if position_mask is not None:
            self._resolve_position_mask(position_mask)
        """Positions on canvas masked for sampling leaf positions."""
        self._unpack_parameters()

    leaf_shape_kw: dict[str, list[str]] = {
        "circular": ["area"],
        "ellipsoid": ["area", "aspect_ratio", "orientation"],
        "rectangular": ["area", "aspect_ratio", "orientation"],
        "polygon": ["area", "n_vertices"],
    }
    """Dictionary connecting keys to respective list of shape parameters."""

    def _resolve_dependencies(self) -> list[str]:
        """Resolve model parameter dependencies.

        Raises:
            ValueError:
                Error on circular or unresolvable dependencies.

        Returns:
            list[str]:
                Sorted list of parameters.
        """
        dependencies = {param: set() for param in self.shape_param_distributions}
        for param, dist in self.shape_param_distributions.items():
            dist_params = next(iter(dist.values()))
            for dist_param in dist_params.values():
                if isinstance(dist_param, dict) and "from" in dist_param:
                    dependencies[param].union(set(dist_param["from"]))

        resolved, ordered_params = set(["x_pos", "y_pos"]), ["x_pos", "y_pos"]
        while dependencies:
            ready = [param for param, dep in dependencies.items() if dep <= resolved]
            if not ready:
                raise ValueError("Circular or unresolved dependencies detected")
            ordered_params.extend(ready)
            resolved.update(ready)
            for param in ready:
                dependencies.pop(param)
        return ordered_params

    def _unpack_parameters(self) -> None:
        """Generate list of model parameters and distribution instances to sample from.

        Raises:
            ValueError:
                Provided parameters don't match provided leaf shape.
        """
        self.params = list(self.shape_param_distributions.keys())
        if set(self.params) != set(self.leaf_shape_kw[self.leaf_shape]):
            raise ValueError(
                f"Model with {self.leaf_shape} shapes expects parameters: "
                f"{self.leaf_shape_kw[self.leaf_shape]} but received {self.params}"
            )
        sampling_box = bounding_box(self.position_mask, 1)
        self.distributions = {
            "x_pos": torch.distributions.uniform.Uniform(
                sampling_box[1], sampling_box[3]
            ),
            "y_pos": torch.distributions.uniform.Uniform(
                sampling_box[0], sampling_box[2]
            ),
        }
        for param, dist_dict in self.shape_param_distributions.items():
            if len(dist_dict) != 1:
                raise ValueError(
                    f"Distribution dictionary for {param} contains "
                    f"{len(dist_dict)} keys, but 1 is required."
                )
            dist_name = list(dist_dict.keys())[0]
            dist_class = dist_kw[dist_name]
            hyper_params = dist_dict[dist_name].copy()
            if set(hyper_params.keys()) != dist_params[dist_name]:
                raise ValueError(
                    f"Distribution dictionary for {param} with distribution "
                    f"{dist_name} expects parameters: {dist_params[dist_name]} "
                    f"but received {set(hyper_params.keys())}"
                )
            if any([isinstance(p, dict) for p in hyper_params.values()]):
                self.distributions[param] = None
            else:
                self.distributions[param] = dist_class(**hyper_params)
        self.params = list(self.distributions.keys())

    def _sample_parameters(self) -> dict[str, torch.Tensor]:
        """Draw a sample from the model distributions.

        Returns:
            dict[str, torch.Tensor]:
                Sample for each model parameter.
        """
        with self.device:
            samples = {}
            params = self._resolve_dependencies()
            for param in params:
                dist = self.distributions[param]
                if dist is None:
                    dist_dict = self.shape_param_distributions[param]
                    if len(dist_dict) != 1:
                        raise ValueError(
                            f"Distribution dictionary for {param} contains "
                            f"{len(dist_dict)} keys, but 1 is required."
                        )
                    dist_name = list(dist_dict.keys())[0]
                    dist_class = dist_kw[dist_name]
                    hyper_params = dist_dict[dist_name].copy()
                    if set(hyper_params.keys()) != dist_params[dist_name]:
                        raise ValueError(
                            f"Distribution dictionary for {param} with distribution "
                            f"{dist_name} expects parameters: {dist_params[dist_name]} "
                            f"but received {set(hyper_params.keys())}"
                        )
                    for idx, hyper_param in hyper_params.items():
                        if isinstance(hyper_param, dict):
                            if isinstance(hyper_param["from"], str):
                                hyper_params[idx] = torch.tensor(
                                    hyper_param["fn"](samples[hyper_param["from"]])
                                )
                            else:
                                hyper_params[idx] = torch.tensor(
                                    hyper_param["fn"](
                                        {
                                            key: samples[key]
                                            for key in hyper_param["from"]
                                        }
                                    )
                                )
                    dist = dist_class(**hyper_params)
                samples[param] = dist.sample()
            return samples

    def _resolve_position_mask(
        self, position_mask: torch.Tensor | np.ndarray | dict
    ) -> None:
        """Resolve position mask from tensor or geometric specification.

        Args:
            position_mask (torch.Tensor | np.ndarray | dict):
                Boolean tensor or array with allowed sampling positions or
                dictionary specifying a leaf shape and its parameters.

        Raises:
            ValueError: Position mask dict must contain 'shape' and 'params'.
            ValueError: Check for unsupported leaf shapes in dict.
            TypeError: Check argument type.
            ValueError: Check that position mask has same size as canvas.
            ValueError: Check that there are allowed positions.
        """

        # Case 1: already a tensor
        if isinstance(position_mask, (torch.Tensor, np.ndarray)):
            position_mask = torch.as_tensor(
                position_mask, device=self.device, dtype=torch.int
            )
            self.position_mask = position_mask.to(device=self.device, dtype=int)

        # Case 2: geometric specification
        elif isinstance(position_mask, dict):
            if "shape" not in position_mask or "params" not in position_mask:
                raise ValueError(
                    "Position mask dict must contain 'shape' and 'params'."
                )

            # default position to image center
            default_params = {
                "x_pos": self.image_shape[1] // 2,
                "y_pos": self.image_shape[0] // 2,
            }
            params = {**default_params, **position_mask["params"]}

            leaf_shape = position_mask["shape"]
            params = {
                k: torch.as_tensor(v, device=self.device) for k, v in params.items()
            }

            if leaf_shape not in leaf_mask_kw:
                raise ValueError(f"Unknown shape '{leaf_shape}' for position mask.")

            generate_mask = leaf_mask_kw[leaf_shape]
            mask = generate_mask((self.X, self.Y), params)
            self.position_mask = mask.to(device=self.device, dtype=int)

        else:
            raise TypeError(
                "position_mask must be a torch.Tensor, np.array, dict, or None."
            )

        if self.position_mask.shape != self.image_shape:
            raise ValueError("Position mask must match image size.")

        if not torch.any(self.position_mask):
            raise ValueError("Position mask is all zeros. No valid sampling positions.")

    def generate_segmentation(self) -> tuple[pd.DataFrame, torch.Tensor]:
        """Generate a dead leaves segmentation from the model.

        Returns:
            tuple[pd.DataFrame, torch.Tensor]:
                DataFrame listing the parameters of all generated leaves, along
                with an segmentation map assigning each image location to a leaf.
        """
        leaves_params = []
        segmentation_map = torch.zeros(self.image_shape, device=self.device, dtype=int)
        leaf_idx = 1

        while torch.any((segmentation_map == 0) & (self.position_mask == 1)):
            params = self._sample_parameters()
            leaf_mask = self.generate_leaf_mask((self.X, self.Y), params)
            mask = leaf_mask & (segmentation_map == 0)
            if (mask.sum() > 0) & self.position_mask[
                params["y_pos"].to(int), params["x_pos"].to(int)
            ]:
                segmentation_map[mask] = leaf_idx
                leaves_params.append(params)
                leaf_idx += 1
            if (self.n_sample is not None) and leaf_idx >= self.n_sample:
                break

        leaf_table = pd.DataFrame(leaves_params, columns=self.params)
        leaf_table["leaf_idx"] = torch.tensor(range(leaf_idx - 1)) + 1
        leaf_table["leaf_shape"] = self.leaf_shape
        return leaf_table, segmentation_map


class LeafAppearanceSampler:
    """Setup color and texture model for a dead leaves partition.

    Args:
        leaf_table (pd.DataFrame):
            Dataframe of leaves and their parameters.
        color_param_distributions (dict[str, dict[str, dict[str, float]]])
            Color parameters and their distribution setup.
        texture_param_distributions (dict[str, dict[str, dict[str, float]]], optional):
            Texture parameters and their distribution setup. Defaults to constant 0,
            i.e. no texture.
        background_color (torch.Tensor, optional):
            For images which are not fully covered (due to a position mask or sparse
            sampling) one can set a RGB background color. If None the back ground will
            be black. Defaults to None.
    """

    def __init__(
        self,
        leaf_table: pd.DataFrame,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ):
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )
        """Chosen compute backend."""
        self.leaf_table: pd.DataFrame = leaf_table.copy()
        """Dataframe of leaves and their parameters."""
        self.n_leaves = len(self.leaf_table)

    # -------------------------------------
    # Color
    # -------------------------------------
    def _unpack_color(self) -> None:
        """Generate color distribution instances to sample from."""
        with self.device:
            self.color_distributions = {}
            for param, dist_dict in self.color_param_distributions.items():
                if len(dist_dict) != 1:
                    raise ValueError(
                        f"Distribution dictionary for {param} contains "
                        f"{len(dist_dict)} keys, but 1 is required."
                    )
                dist_name = list(dist_dict.keys())[0]
                dist_class = dist_kw[dist_name]
                hyper_params = dist_dict[dist_name].copy()
                if set(hyper_params.keys()) != dist_params[dist_name]:
                    raise ValueError(
                        f"Distribution dictionary for {param} with distribution "
                        f"{dist_name} expects parameters: {dist_params[dist_name]} "
                        f"but received {set(hyper_params.keys())}"
                    )
                for idx, hyper_param in hyper_params.items():
                    if isinstance(hyper_param, dict):
                        hyper_params[idx] = torch.tensor(
                            hyper_param["fn"](self.leaf_table[hyper_param["from"]])
                        )
                self.color_distributions[param] = dist_class(**hyper_params)

    def _sample_grayscale_colors(self) -> torch.Tensor:
        """Sample a grayscale color for each leaf.

        Returns:
            torch.Tensor:
                Color values.
        """
        with self.device:
            for dist in self.color_distributions.values():
                if len(dist.batch_shape) == 0:
                    colors = dist.sample((self.n_leaves, 1))
                else:
                    colors = dist.sample().expand(1, -1).permute(1, 0)
        return colors

    def _sample_3d_colors(self) -> torch.Tensor:
        """Sample a 3d color for each leaf.

        Returns:
            torch.Tensor:
                Color values.
        """
        with self.device:
            colors = {}
            for param, dist in self.color_distributions.items():
                if len(dist.batch_shape) == 0:
                    colors[param] = dist.sample((self.n_leaves,))
                else:
                    colors[param] = dist.sample()
            color_tensor = torch.stack(
                [colors[param] for param in self.color_space], dim=1
            )
        return color_tensor.cpu()

    def _sample_colors_from_images(self) -> torch.Tensor:
        """Sample a pixel value for each leaf from a random image in folder.

        Returns:
            torch.Tensor:
                Color values.
        """
        with self.device:
            for _, dist in self.color_distributions.items():
                image_path = dist.sample()[0]
            image = pil_to_tensor(PIL.Image.open(image_path)) / 255
            image_vector = image.reshape((3, -1))
            idx = torch.multinomial(
                torch.ones(image_vector.shape[-1]), self.n_leaves, replacement=True
            )
            color_tensor = image_vector[:, idx]
        return color_tensor.permute(1, 0)

    def sample_color(
        self,
        color_param_distributions: dict[str, dict[str, dict[str, float]]],
    ) -> torch.Tensor:
        """Sample leaf colors according to the configured color space.

        Returns:
            pd.DataFrame:
                Updated DataFrame with color parameters added to each leaf
        """
        self.color_param_distributions: dict[str, dict[str, dict[str, float]]] = (
            color_param_distributions
        )
        self.color_space = color_spaces[
            tuple(sorted(list(self.color_param_distributions.keys())))
        ]
        self._unpack_color()

        color_columns = [f"color_{k}" for k in self.color_distributions]
        color_columns_rgb = ["color_R", "color_G", "color_B"]

        if self.color_space == ("R", "G", "B"):
            color = self._sample_3d_colors()

        elif self.color_space == ("H", "S", "V"):
            color = self._sample_3d_colors()
            self.leaf_table[color_columns_rgb] = hsv_to_rgb(color)

        elif self.color_space == "gray":
            color = self._sample_grayscale_colors()
            self.leaf_table[color_columns_rgb] = color.expand(-1, 3)

        elif self.color_space == "source":
            color = self._sample_colors_from_images()
            color_columns = color_columns_rgb

        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")

        self.leaf_table[color_columns] = color
        return self.leaf_table

    # -------------------------------------
    # Texture
    # -------------------------------------
    def _unpack_texture(self) -> None:
        """Generate texture distribution instances to sample from."""
        with self.device:
            self.texture_distributions = {}

            for param, dist_dict in self.texture_param_distributions.items():
                dist_name, hyper_params = next(iter(dist_dict.items()))
                dist_class = dist_kw[dist_name]

                resolved_params = {}
                if param == "alpha":
                    resolved_params["alpha"] = dist_class(**hyper_params).sample(
                        (self.n_leaves, 1)
                    )
                else:
                    for key, value in hyper_params.items():
                        if isinstance(value, dict):
                            sub_dist_name, sub_params = next(iter(value.items()))
                            sub_dist_class = dist_kw[sub_dist_name]
                            resolved_params[key] = sub_dist_class(**sub_params).sample(
                                (self.n_leaves,)
                            )
                        elif isinstance(value, str):
                            resolved_params[key] = dist_class(value).sample(
                                self.n_leaves
                            )
                        elif isinstance(value, (int, float)):
                            resolved_params[key] = value

                self.texture_distributions[param] = {
                    "dist_name": dist_name,
                    "dist_class": dist_class,
                    "params": resolved_params,
                }

    def _sample_texture_parameters(self) -> torch.Tensor:
        """
        Materialize per-leaf texture parameters and distribution metadata.

        Returns:
            pd.DataFrame
                One row per leaf with resolved texture parameters for all channels.
        """
        with self.device:
            rows = []
            for leaf_idx in self.leaf_table.leaf_idx:
                row = {"leaf_idx": leaf_idx}

                for channel_name, channel_cfg in self.texture_distributions.items():
                    dist_name = channel_cfg["dist_name"]
                    params = channel_cfg["params"]

                    # distribution metadata
                    row[f"texture_{channel_name}_dist"] = dist_name

                    # per-parameter values
                    for param, value in params.items():
                        if isinstance(value, torch.Tensor):
                            row[f"texture_{channel_name}_{param}"] = value[
                                leaf_idx - 1
                            ].item()
                        elif isinstance(value, list):
                            row[f"texture_{channel_name}_{param}"] = value[leaf_idx - 1]
                        else:
                            row[f"texture_{channel_name}_{param}"] = value

                rows.append(row)
            return pd.DataFrame(rows)

    def sample_texture(
        self,
        texture_param_distributions: dict[
            str, dict[str, dict[str, float | dict[str, dict[str, float]]]]
        ],
    ) -> torch.Tensor:
        """Sample leaf textures according to the configured color space.

        pd.DataFrame:
            Updated DataFrame with texture parameters added to each leaf
        """
        self.texture_param_distributions: dict[
            str, dict[str, dict[str, float | dict[str, dict[str, float]]]]
        ] = texture_param_distributions
        self.texture_space = color_spaces[
            tuple(sorted(list(self.texture_param_distributions.keys())))
        ]

        self._unpack_texture()

        if self.texture_space in ("gray", ("R", "G", "B"), ("H", "S", "V"), "source"):
            df = self._sample_texture_parameters()

        else:
            raise ValueError(f"Unsupported texture space: {self.texture_space}")

        self.leaf_table = self.leaf_table.merge(df, on="leaf_idx", how="left")
        return self.leaf_table


class ImageRenderer:
    """Setup color and texture model for a dead leaves partition.

    Args:
        leaf_table (pd.DataFrame):
            Dataframe of leaves and their parameters.
        segmentation_map (torch.Tensor):
            Partition which assigns image locations to a leaves
        background_color (torch.Tensor, optional):
            For images which are not fully covered (due to a position mask or sparse
            sampling) one can set a RGB background color. If None the back ground will
            be black. Defaults to None.
    """

    def __init__(
        self,
        leaf_table: pd.DataFrame,
        segmentation_map: torch.Tensor | None = None,
        image_shape: tuple[int, int] | None = None,
        background_color: torch.Tensor | None = None,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ):
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )
        """Chosen compute backend."""
        self.background_color: torch.Tensor | None = background_color
        """Color for pixels not belonging to any leaf."""
        if isinstance(background_color, torch.Tensor):
            self.background_color = self.background_color.to(device=self.device)
        self.leaf_table: pd.DataFrame = leaf_table
        """Dataframe of leaves and their parameters."""
        self.segmentation_map: torch.Tensor = segmentation_map
        """Partition of the image area."""
        self.image_shape: torch.Size = image_shape
        """Height (y, M) and width (x, N) of the canvas."""
        self._resolve_image_shape()
        if segmentation_map is None:
            self._generate_segmentation_map()
        self._infer_texture_space()

    def _resolve_image_shape(self) -> None:
        """Resolve image shape from segmentation_map or explicit image_shape."""
        if self.segmentation_map is not None:
            if (
                self.image_shape is not None
                and self.segmentation_map.shape != self.image_shape
            ):
                raise ValueError(
                    f"Segmentation map shape {self.segmentation_map.shape} "
                    f"does not match provided image_shape {self.image_shape}"
                )
            self.image_shape = self.segmentation_map.shape
        elif self.image_shape is None:
            raise ValueError(
                "Must provide at least one of 'segmentation_map' or 'image_shape'."
            )

    def _generate_segmentation_map(self) -> None:
        """Generate segmentation map if None is provided"""
        topology = LeafTopology(image_shape=self.image_shape)
        self.segmentation_map = topology.segmentation_map_from_table(self.leaf_table)

    def _infer_texture_space(self) -> None:
        """Infer which color space the texture parameters are defined in"""
        keys = sorted(
            col.removeprefix("texture_").split("_")[0]
            for col in self.leaf_table.columns
            if col.startswith("texture_") and col.endswith("_dist")
        )
        self.texture_space = color_spaces.get(tuple(keys), None)

    def _get_texture_param_columns(self, channel: str) -> list[str]:
        """Get all columns which contain texture parameters from leaf_table"""
        prefix = f"texture_{channel}_"
        return [
            c
            for c in self.leaf_table.columns
            if c.startswith(prefix) and not c.endswith("_dist")
        ]

    def _get_leaf_texture_params(self, leaf_row, channel: str) -> dict:
        """Get values from columns with texture parameters in leaf_table"""
        prefix = f"texture_{channel}_"
        hyper_params = {}

        for col in self._get_texture_param_columns(channel):
            param_name = col[len(prefix) :]
            hyper_params[param_name] = leaf_row[col]

        return hyper_params

    def _generate_leafwise_texture_1d(self, channel):
        """
        Generate grayscale texture from sampled distributions.

        Returns:
            torch.Tensor
                Texture image of shape (H, W).
        """
        texture = torch.zeros(self.image_shape, device=self.device)
        for _, leaf in self.leaf_table.iterrows():
            dist_name = leaf[f"texture_{channel}_dist"]
            dist_class = dist_kw[dist_name]
            hyper_params = self._get_leaf_texture_params(leaf, channel)
            leaf_texture = dist_class(**hyper_params).sample(self.image_shape)

            mask = self.segmentation_map == leaf["leaf_idx"]
            texture[mask] = leaf_texture[mask]
        return texture

    def _generate_leafwise_texture_from_source(self) -> torch.Tensor:
        """
        Generate grayscale texture from image sources stored in leaf_table.

        Returns:
            torch.Tensor
                Texture image of shape (H, W).
        """
        H, W = self.image_shape
        texture = torch.zeros((H, W), device=self.device)

        X, Y = torch.meshgrid(
            torch.arange(W, device=self.device),
            torch.arange(H, device=self.device),
            indexing="xy",
        )

        for i, row in self.leaf_table.iterrows():
            leaf_idx = row.leaf_idx
            leaf_mask = self.segmentation_map == leaf_idx

            unoccluded_mask = leaf_mask_kw[row["shape"]]((X, Y), row)
            top, left, bottom, right = bounding_box(unoccluded_mask, 1)
            vh, vw = bottom - top, right - left
            if vh <= 0 or vw <= 0:
                continue

            patch = (
                pil_to_tensor(
                    PIL.Image.open(row["texture_source_dir"])
                    .convert("L")
                    .resize((vw, vh))
                ).to(self.device)
                / 255.0
            )
            texture_patch = torch.zeros((H, W), device=self.device)
            texture_patch[top:bottom, left:right] = patch
            texture += leaf_mask * row["texture_alpha_alpha"] * texture_patch
        return texture

    def _generate_leafwise_texture(self) -> torch.Tensor:
        """Generate a per-pixel texture image from leafwise texture parameters.

        Returns:
            torch.Tensor
                Texture image of shape (H, W, 3).
        """
        H, W = self.image_shape
        texture = torch.zeros((H, W, 3), device=self.device)

        if self.texture_space is None:
            return texture

        if self.texture_space == "gray":
            gray = self._generate_leafwise_texture_1d("gray")
            return gray.unsqueeze(-1).expand(-1, -1, 3)

        if self.texture_space in [("R", "G", "B"), ("H", "S", "V")]:
            for channel in self.texture_space:
                idx = {"R": 0, "G": 1, "B": 2, "H": 0, "S": 1, "V": 2}[channel]
                texture[:, :, idx] = self._generate_leafwise_texture_1d(channel)

            if self.texture_space == ("H", "S", "V"):
                texture = torch.tensor(hsv_to_rgb(texture.cpu()), device=self.device)
            return texture

        if self.texture_space == "source":
            gray = self._generate_leafwise_texture_from_source()
            texture = gray.unsqueeze(-1).expand(-1, -1, 3)
            return texture

    def render_image(self) -> torch.Tensor:
        """Generate a dead leaves image.

        Returns:
            torch.Tensor:
                Dead leaves image tensor.
        """
        with self.device:
            image = torch.zeros(self.image_shape + (3,), device=self.device)
            colors = torch.tensor(
                self.leaf_table[["color_R", "color_G", "color_B"]].to_numpy(),
                dtype=torch.float32,
                device=self.device,
            )
            texture = self._generate_leafwise_texture()
            for leaf_idx in self.leaf_table.leaf_idx:
                image[self.segmentation_map == leaf_idx] = torch.clip(
                    colors[leaf_idx - 1] + texture[self.segmentation_map == leaf_idx],
                    0,
                    1,
                )
            if self.background_color is not None:
                image[self.segmentation_map == 0] = self.background_color
            return image

    def apply_image_noise(
        self, image: torch.Tensor, noise: dict[str, dict | torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply global noise to an image.

        Args:
            image (torch.Tensor):
                Input image tensor of shape (H, W, C), values in [0,1].
            noise (dict):
                Dictionary or tensor/array specifying noise per channel.
                Examples:
                - Distribution-based: {"gray": {"normal": {"loc": 0, "scale": 0.1}}}
                - Array-based: {"R": torch.randn(H, W) * 0.05}

        Returns:
            torch.Tensor:
                Image with noise applied, clipped to [0,1].
        """
        noisy_image = image.clone()
        H, W, C = noisy_image.shape

        for channel, spec in noise.items():
            # Case 1: distribution-based noise
            if isinstance(spec, dict):
                dist_name = next(iter(spec.keys()))
                dist_class = dist_kw[dist_name]
                channel_noise = (
                    dist_class(**spec[dist_name]).sample((H, W)).to(self.device)
                )

            # Case 2: array-based noise
            elif isinstance(spec, (torch.Tensor, np.ndarray)):
                channel_noise = torch.as_tensor(
                    spec, device=self.device, dtype=torch.float32
                )
                if channel_noise.shape != (H, W):
                    raise ValueError(
                        f"Noise for channel {channel} must have shape {(H, W)}"
                    )

            else:
                raise TypeError(
                    f"Noise spec for channel {channel} must be a dict or tensor/array"
                )

            if channel == "gray":
                for i in range(3):
                    noisy_image += channel_noise.unsqueeze(-1)

            elif channel in ["R", "G", "B"]:
                idx = {"R": 0, "G": 1, "B": 2}[channel]
                noisy_image[:, :, idx] += channel_noise

            else:
                raise ValueError(
                    f"Unsupported channel for image-wide texture: '{channel}'. "
                    "Should be 'gray', 'R', 'G', or 'B'."
                )

        return torch.clip(noisy_image, 0, 1)

    def show(self, image: torch.Tensor, figsize: tuple[int, int] | None = None) -> None:
        """Show selected image.

        Args:
            image (torch.Tensor):
                Image to show.
            figsize (tuple[int,int], optional):
                Figure size in inches (width, height). If None size is inferred from
                image size. Defaults to None.
        """
        fig, ax = plt.subplots(figsize=figsize, frameon=False)
        ax.imshow(image.cpu().numpy(), vmax=1, vmin=0)
        fig.tight_layout()
        ax.axis("off")

        plt.show()

    def save(self, image: torch.Tensor, save_to: Path | str) -> None:
        """Save image to path.

        Args:
            image (torch.Tensor):
                Image to save.
            save_to (Path):
                Path to file to save image to.
        """
        plt.imsave(save_to, image.cpu().numpy())

    def animate(self, fps: int = 10, save_to: Path = None) -> animation.FuncAnimation:
        """Generate animation of dead leaves partition generation.

        Args:
            fps (int, optional):
                Frames per second of animation. In each frame a new
                leaf is sampled. Defaults to 10.
            save_to (Path, optional):
                Path to file to save animation to. If None the
                animation will not be saved. Defaults to None.

        Returns:
            animation.FuncAnimation:
                Animation of partition generation.
        """
        fig, ax = plt.subplots(frameon=False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(0, self.image_shape[1])
        ax.set_ylim(0, self.image_shape[0])

        def add_leaf(frame):
            leaf = self.leaf_table.iloc[frame]
            circle = plt.Circle(
                (leaf.y_pos.cpu(), leaf.x_pos.cpu()),
                leaf.radius.cpu(),
                zorder=-frame,
                edgecolor="black",
                facecolor="lightgray",
            )
            ax.add_patch(circle)
            return ax

        dl_animation = animation.FuncAnimation(
            fig,
            add_leaf,
            frames=len(self.leaf_table),
            interval=1000 / fps,
            repeat=False,
        )

        if save_to:
            FFwriter = animation.FFMpegWriter(fps=fps)
            dl_animation.save(save_to, writer=FFwriter)

        return dl_animation


class LeafTopology:
    """
    Topology-level operations on leaf tables to
    - Construct segmentation maps
    - Merge and relabel leaf tables
    - Manage leaf identities (leaf_idx)
    """

    def __init__(
        self,
        image_shape: tuple[int, int] | None = None,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ):
        self.image_shape: tuple[int, int] | None = image_shape
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )

    def _validate_geometry(self, leaf_table: pd.DataFrame) -> None:
        required = {"x_pos", "y_pos", "leaf_shape", "leaf_idx", "area"}
        missing = required - set(leaf_table.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _cull_outside_canvas(
        self,
        leaf_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove leaves whose centers are outside the canvas.
        """
        H, W = self.image_shape
        return leaf_table[
            (leaf_table.x_pos >= 0)
            & (leaf_table.x_pos < W)
            & (leaf_table.y_pos >= 0)
            & (leaf_table.y_pos < H)
        ].copy()

    def segmentation_map_from_table(
        self,
        leaf_table: pd.DataFrame,
    ) -> torch.Tensor:
        """
        Construct a segmentation map from a leaf table.

        Args:
            leaf_table (pd.DataFrame):
                Dataframe of leaves and their parameters.

        Returns:
            torch.Tensor:
                Segementation map which assigns each image location to a leaf.

        """
        H, W = self.image_shape
        segmentation_map = torch.zeros((H, W), device=self.device, dtype=torch.int64)

        X, Y = torch.meshgrid(
            torch.arange(W, device=self.device),
            torch.arange(H, device=self.device),
            indexing="xy",
        )

        self._validate_geometry(leaf_table)
        leaf_table = self._cull_outside_canvas(leaf_table)
        for _, row in leaf_table.iterrows():
            generate_leaf_mask = leaf_mask_kw[row["leaf_shape"]]
            leaf_mask = generate_leaf_mask((X, Y), row.to_dict())
            mask = leaf_mask & (segmentation_map == 0)
            segmentation_map[mask] = int(row["leaf_idx"])
        return segmentation_map

    def merge_leaf_tables(self, *leaf_tables: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple leaf tables and assign fresh leaf_idx.
        """
        merged = pd.concat(leaf_tables, ignore_index=True)
        merged = merged.copy()
        merged["leaf_idx"] = np.arange(1, len(merged) + 1)
        return merged

    def reindex_by_group(
        self,
        leaf_table: pd.DataFrame,
        groupby: str,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Reassign leaf_idx within groups.
        """
        rng = np.random.default_rng(seed)
        out = []
        start = 1

        for _, group in leaf_table.groupby(groupby, sort=False):
            g = group.copy()
            if shuffle:
                g = g.sample(frac=1, random_state=rng.integers(1e9))
            g["leaf_idx"] = np.arange(start, start + len(g))
            start += len(g)
            out.append(g)
        leaf_table = pd.concat(out, ignore_index=True)
        leaf_table = leaf_table.sort_values(by="leaf_idx", ascending=True)
        return leaf_table

    def shuffle_index_within_group(
        self,
        leaf_table: pd.DataFrame,
        groupby: str | list[str],
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Shuffle rows within groups and reassign a contiguous index column.

        Args:
            table (pd.DataFrame):
                Input leaf table.
            groupby (str or list[str]):
                Column(s) defining groups (e.g. "type").
            seed (int, optional):
                Random seed for reproducibility.

        Returns:
            pd.DataFrame:
                Table with reassigned leaf_idx.
        """
        rng = np.random.default_rng(seed)
        out = []
        start = 1

        for _, group in leaf_table.groupby(groupby, sort=False):
            g = group.copy().sample(
                frac=1, random_state=rng.integers(np.iinfo(np.int32).max)
            )
            g["leaf_idx"] = np.arange(start, start + len(g))
            start += len(g)
            out.append(g)
        leaf_table = pd.concat(out, ignore_index=True)
        leaf_table = leaf_table.sort_values(by="leaf_idx", ascending=True)
        return leaf_table

    def randomize_index(
        leaf_table: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Randomly reassign a global index while preserving all row attributes.

        This shuffles depth ordering across all leaves, including across semantic
        groups (e.g. target vs background), while keeping group labels intact.

        Args:
            table (pd.DataFrame):
                Input leaf table.
            seed (int, optional):
                Random seed for reproducibility.

        Returns:
            pd.DataFrame:
                Table with randomized index_col.
        """
        rng = np.random.default_rng(seed)
        leaf_table = leaf_table.copy()
        leaf_table["leaf_idx"] = rng.permutation(np.arange(1, len(leaf_table) + 1))
        leaf_table = leaf_table.sort_values(by="leaf_idx", ascending=True)
        return leaf_table
