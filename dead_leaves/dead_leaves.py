from .distributions import PowerLaw, Constant, Cosine, ExpCosine, Image
from .leaf_masks import circular, rectangular, ellipsoid, regular_polygon
from .utils import choose_compute_backend, bounding_box
from typing import Literal, Callable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import torch
import pandas as pd
import PIL.Image
from pathlib import Path
from torchvision.transforms.functional import pil_to_tensor

import warnings

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
        shape (Literal["circular", "ellipsoid", "rectangular", "polygon"]):
            Shape of leaves.
        param_distributions (dict[str, dict[str, [dict[str,float]]]):
            Shape parameters and their distributions and distribution parameter values.
        size (tuple[int, int]):
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
        shape: Literal["circular", "ellipsoid", "rectangular", "polygon"],
        param_distributions: dict[str, dict[str, dict[str, float]]],
        size: tuple[int, int],
        position_mask: torch.Tensor | None = None,
        n_sample: int | None = None,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ) -> None:
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )
        """Chosen compute backend."""
        self.size: tuple[int, int] = size
        """Height (y, M) and width (x, N) of the canvas."""
        self.position_mask: torch.Tensor = torch.ones(
            self.size, dtype=int, device=self.device
        )
        """Positions on canvas masked for sampling leaf positions."""
        if position_mask is not None:
            if position_mask.shape != size:
                raise ValueError("Position mask needs to match image size.")
            self.position_mask: torch.Tensor = position_mask.to(device=self.device)
        self.n_sample: int | None = n_sample
        """Number of leaves to sample."""
        self.shape: (
            Literal["circular"]
            | Literal["ellipsoid"]
            | Literal["rectangular"]
            | Literal["polygon"]
        ) = shape
        """Shape of the leaves."""
        self.param_distributions: dict[str, dict[str, dict[str, float]]] = (
            param_distributions
        )
        """Shape parameters and their distributions and distribution parameter values."""
        self.X, self.Y = torch.meshgrid(
            torch.arange(self.size[1], device=self.device),
            torch.arange(self.size[0], device=self.device),
            indexing="xy",
        )
        self.generate_leaf_mask: Callable = leaf_mask_kw[shape]
        """Method to generate mask of leaf on canvas."""
        self._configure_parameters()

    shape_kw: dict[str, list[str]] = {
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
        dependencies = {param: set() for param in self.param_distributions}
        for param, dist in self.param_distributions.items():
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

    def _configure_parameters(self) -> None:
        """Generate list of model parameters and distribution instances to sample from.

        Raises:
            ValueError:
                Provided parameters don't match provided shape.
        """
        self.params = list(self.param_distributions.keys())
        if set(self.params) != set(self.shape_kw[self.shape]):
            raise ValueError(
                f"Model with {self.shape} shapes expects parameters: "
                f"{self.shape_kw[self.shape]} but received {self.params}"
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
        for param, dist_dict in self.param_distributions.items():
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
                    dist_dict = self.param_distributions[param]
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

    def generate_instance(self) -> tuple[pd.DataFrame, torch.Tensor]:
        """Generate a dead leaves instance from the model.

        Returns:
            tuple[pd.DataFrame, torch.Tensor]:
                DataFrame listing the parameters of all generated leaves, along
                with an instance map assigning each image location to a leaf.
        """
        leaves_params = []
        instance_map = torch.zeros(self.size, device=self.device, dtype=int)
        leaf_idx = 1

        while torch.any((instance_map == 0) & (self.position_mask == 1)):
            params = self._sample_parameters()
            leaf_mask = self.generate_leaf_mask((self.X, self.Y), params)
            mask = leaf_mask & (instance_map == 0)
            if (mask.sum() > 0) & self.position_mask[
                params["y_pos"].to(int), params["x_pos"].to(int)
            ]:
                instance_map[mask] = leaf_idx
                leaves_params.append(params)
                leaf_idx += 1
            if (self.n_sample is not None) and leaf_idx >= self.n_sample:
                break

        instance_table = pd.DataFrame(leaves_params, columns=self.params)
        instance_table["leaf_idx"] = torch.tensor(range(leaf_idx - 1)) + 1
        instance_table["shape"] = self.shape
        return instance_table, instance_map


class LeafAppearanceSampler:
    """Setup color and texture model for a dead leaves partition.

    Args:
        instance_table (pd.DataFrame):
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
        instance_table: pd.DataFrame,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ):
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )
        """Chosen compute backend."""
        self.instance_table: pd.DataFrame = instance_table.copy()
        """Dataframe of leaves and their parameters."""
        self.n_leaves = len(self.instance_table)

    # -------------------------------------
    # Color
    # -------------------------------------
    def _configure_color(self) -> None:
        """Generate distribution instances to sample from."""
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
                            hyper_param["fn"](self.instance_table[hyper_param["from"]])
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
        
        pd.DataFrame:
            Updated DataFrame with color parameters added to each leaf
        """
        self.color_param_distributions: dict[str, dict[str, dict[str, float]]] = (
            color_param_distributions
        )
        self.color_space = color_spaces[
            tuple(sorted(list(self.color_param_distributions.keys())))
        ]
        self._configure_color()
        
        color_columns = [f"color_{k}" for k in self.color_distributions]
        color_columns_rgb = ["color_R", "color_G", "color_B"]
        
        if self.color_space == ("R", "G", "B"):
            color = self._sample_3d_colors()
        
        elif self.color_space == ("H", "S", "V"):
            color = self._sample_3d_colors()
            self.instance_table[color_columns_rgb] = hsv_to_rgb(color)
    
        elif self.color_space == "gray":
            color = self._sample_grayscale_colors()
            self.instance_table[color_columns_rgb] =  color.expand(-1, 3)
    
        elif self.color_space == "source":
            color = self._sample_colors_from_images()
            color_columns = color_columns_rgb
    
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")

        self.instance_table[color_columns] = color
        return self.instance_table

    # -------------------------------------
    # Texture
    # -------------------------------------    
    def _configure_texture(self) -> None:
        with self.device:
            self.texture_distributions = {}
        
            for param, dist_dict in self.texture_param_distributions.items():
                if len(dist_dict) != 1:
                    raise ValueError(
                        f"Distribution dictionary for {param} must contain exactly one entry."
                    )
    
                dist_name, hyper_params = next(iter(dist_dict.items()))
                dist_class = dist_kw[dist_name]
        
                resolved_params = {}
                for key, value in hyper_params.items():
                    if isinstance(value, dict):
                        sub_dist_name, sub_params = next(iter(value.items()))
                        sub_dist_class = dist_kw[sub_dist_name]
                        resolved_params[key] = sub_dist_class(**sub_params).sample((self.n_leaves,))
                    elif isinstance(value, str):
                        resolved_params[key] = dist_class(value).sample(self.n_leaves)
                    elif isinstance(value, float):
                        resolved_params[key] = value
        
                self.texture_distributions[param] = {
                    "dist_name": dist_name,
                    "dist_class": dist_class,
                    "params": resolved_params,
                }

    def _sample_texture_parameters(self) -> torch.Tensor:
        """
        Materialize per-leaf texture parameters and distribution metadata.
    
        Returns
        -------
        pd.DataFrame
            One row per leaf with resolved texture parameters for all channels.
        """
        with self.device:
            rows = []
            for leaf_idx in self.instance_table.leaf_idx:
                row = {"leaf_idx": leaf_idx}
        
                for channel_name, channel_cfg in self.texture_distributions.items():
                    dist_name = channel_cfg["dist_name"]
                    params = channel_cfg["params"]
        
                    # distribution metadata
                    row[f"texture_{channel_name}_dist"] = dist_name
        
                    # per-parameter values
                    for param, value in params.items():
                        if isinstance(value, torch.Tensor):
                            row[f"texture_{channel_name}_{param}"] = value[leaf_idx - 1].item()
                        elif isinstance(value, list):
                            row[f"texture_{channel_name}_{param}"] = value[leaf_idx - 1]
                        else:
                            row[f"texture_{channel_name}_{param}"] = value
    
                rows.append(row)
            return pd.DataFrame(rows)
    
    def sample_texture(
            self,
            texture_param_distributions: dict[str, dict[str, dict[str, float | dict[str, dict[str, float]]]]],
            ) -> torch.Tensor:
        """Sample leaf textures according to the configured color space.
        
        pd.DataFrame:
            Updated DataFrame with texture parameters added to each leaf
        """
        self.texture_param_distributions: dict[str, dict[str, dict[str, float | dict[str, dict[str, float]]]]] = (
            texture_param_distributions
        )
        self.texture_space = color_spaces[
            tuple(sorted(list(self.texture_param_distributions.keys())))
        ]
        
        self._configure_texture()
        
        if self.texture_space in ("gray", ("R", "G", "B"), ("H", "S", "V"), "source"):
            df = self._sample_texture_parameters()
    
        else:
            raise ValueError(f"Unsupported texture space: {self.texture_space}")
        
        self.instance_table = self.instance_table.merge(df, on="leaf_idx", how="left")
        return self.instance_table


class ImageRenderer:
    """Setup color and texture model for a dead leaves partition.

    Args:
        instance_table (pd.DataFrame):
            Dataframe of leaves and their parameters.
        instance_map (torch.Tensor):
            Partition which assigns image locations to a leaves
        background_color (torch.Tensor, optional):
            For images which are not fully covered (due to a position mask or sparse
            sampling) one can set a RGB background color. If None the back ground will
            be black. Defaults to None.
    """

    def __init__(
        self,
        instance_table: pd.DataFrame,
        instance_map: torch.Tensor,
        background_color: torch.Tensor | None = None,
        device: Literal["cuda", "mps", "cpu"] | None = None,
    ):
        self.device: torch.device = (
            torch.device(device) if device else choose_compute_backend()
        )
        """Chosen compute backend."""
        self.size: torch.Size = instance_map.shape
        """Height (y, M) and width (x, N) of the canvas."""
        self.background_color: torch.Tensor | None = background_color
        """Color for pixels not belonging to any leaf."""
        if isinstance(background_color, torch.Tensor):
            self.background_color = self.background_color.to(device=self.device)
        self.instance_table: pd.DataFrame = instance_table
        """Dataframe of leaves and their parameters."""
        self.instance_map: torch.Tensor = instance_map
        """Partition of the image area."""
    
    def _render_texture(self) -> torch.Tensor:
        texture = 1
        
        if texture == ("H", "S", "V"):
            texture = torch.Tensor(hsv_to_rgb(texture.cpu())).to(self.device)
        return texture

    def render_image(self) -> torch.Tensor:
        """Generate a dead leaves image.

        Returns:
            torch.Tensor:
                Dead leaves image tensor.
        """
        with self.device:
            image = torch.zeros(self.size + (3,), device=self.device)
            colors = torch.tensor(
                self.instance_table[["color_R", "color_G", "color_B"]].to_numpy(),
                dtype=torch.float32,
                device=self.device
            )
            # texture = self._render_texture()
            texture = torch.zeros(self.size + (3,), device=self.device)
            for leaf_idx in self.instance_table.leaf_idx:
                image[self.instance_map == leaf_idx] = torch.clip(
                    colors[leaf_idx - 1] + texture[self.instance_map == leaf_idx], 0, 1
                )
            if self.background_color is not None:
                image[self.instance_map == 0] = self.background_color
            return image
    
    def apply_image_noise():
        image = 1
        return image

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
        ax.set_xlim(0, self.size[1])
        ax.set_ylim(0, self.size[0])

        def add_leaf(frame):
            leaf = self.instance_table.iloc[frame]
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
            fig, add_leaf, frames=len(self.instance_table), interval=1000 / fps, repeat=False
        )

        if save_to:
            FFwriter = animation.FFMpegWriter(fps=fps)
            dl_animation.save(save_to, writer=FFwriter)

        return dl_animation
