from .distributions import PowerLaw, Constant, Cosine, ExpCosine, Image
from .utils import choose_compute_backend
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import torch
import pandas as pd
import PIL.Image
from pathlib import Path
from torchvision.transforms.functional import pil_to_tensor

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

dist_kw = {
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


class DeadLeavesModel:
    """Setup a dead leaves model

    Args:
        - shape (str): Shape of leaves.
        - param_distributions (dict[str, dict[str, dict[str, float]]]):
            Shape parameters and their distributions and distribution parameter values.
        - size (tuple[int,int]): Hight (y, M) and width (x, N) of the area to be partitioned.
        - position_mask (tensor): Boolean tensor containing allowed leaf positions to create
            images with different shapes.
        - n_sample (optional, int): Number of leaves to sample. If None the sampling
            will stop when the full area is partitioned. Defaults to None.
        - device: Torch device to use, either cuda or cpu.
        - X, Y (tensor):
        - generate_leaf_mask (callable): Function to generate mask of given shape
            from parameters.
        - params (list[str]): List of shape parameters.
        - distributions (dict[str, Distribution]): Leaf parameters and
            their distributions.

    Methods:
        - model: Generate self.params and self.distributions based on initialization.
        - sample_parameters: Sample values from self.distributions.
        - sample_partition: Generate dead leaves partition by iteratively sampling
            leaves based on the model.
        - _circular_leaf_mask: Generate circular leaf mask from position, and area.
        - _rectangular_leaf_mask: Generate rectangular leaf mask from position, area,
            orientation, and aspect ratio.
        - _ellipsoid_leaf_mask: Generate ellipsoidal leaf mask from position, area,
            orientation, and aspect ratio.
        - _regular_polygon_leaf_mask: Generate regular polygon leaf mask from position,
            area, and number of vertices.
    """

    def __init__(
        self,
        shape: Literal["circular", "ellipsoid", "rectangular", "polygon"],
        param_distributions: dict[str, dict[str, dict[str, float]]],
        size: tuple[int, int],
        device: Literal["cuda", "mps", "cpu"] | None = None,
        position_mask: torch.Tensor | None = None,
        n_sample: int | None = None,
    ) -> None:
        self.device = torch.device(device) if device else choose_compute_backend()
        self.size = size
        if position_mask is not None:
            if position_mask.shape != size:
                raise ValueError("Position mask needs to match image size.")
            self.position_mask = position_mask
        else:
            self.position_mask = torch.ones(self.size, dtype=int, device=self.device)
        self.n_sample = n_sample
        self.shape = shape
        self.param_distributions = param_distributions
        self.X, self.Y = torch.meshgrid(
            torch.arange(self.size[1], device=self.device),
            torch.arange(self.size[0], device=self.device),
            indexing="xy",
        )
        leaf_mask = {
            "circular": self._circular_leaf_mask,
            "ellipsoid": self._ellipsoid_leaf_mask,
            "rectangular": self._rectangular_leaf_mask,
            "polygon": self._regular_polygon_leaf_mask,
        }
        self.generate_leaf_mask = leaf_mask[shape]
        self.model()

    shape_kw = {
        "circular": ["area"],
        "ellipsoid": ["area", "aspect_ratio", "orientation"],
        "rectangular": ["area", "aspect_ratio", "orientation"],
        "polygon": ["area", "n_vertices"],
    }

    def model(self) -> None:
        """Generate list of model parameters and distribution instances to sample from.

        Raises:
            ValueError: Provided parameters don't match provided shape.
        """
        self.params = list(self.param_distributions.keys())
        if set(self.params) != set(self.shape_kw[self.shape]):
            raise ValueError(
                f"Model with {self.shape} shapes expects parameters: "
                f"{self.shape_kw[self.shape]} but received {self.params}"
            )
        self.distributions = {
            "x_pos": torch.distributions.uniform.Uniform(0, self.size[1]),
            "y_pos": torch.distributions.uniform.Uniform(0, self.size[0]),
        }
        for param, dist_dict in self.param_distributions.items():
            dist_name = list(dist_dict.keys())[0]
            dist_class = dist_kw[dist_name]
            hyper_params_dict = dist_dict[dist_name]
            hyper_params = list(hyper_params_dict.values())
            self.distributions[param] = dist_class(*hyper_params)
        self.params = list(self.distributions.keys())

    def sample_parameters(self) -> dict[str, torch.Tensor]:
        """Draw a sample from the model distributions.

        Returns:
            dict[str, tensor]: Sample for each model parameter.
        """
        with self.device:
            samples = {}
            for param, dist in self.distributions.items():
                samples[param] = dist.sample()
            return samples

    def sample_partition(self) -> tuple[pd.DataFrame, torch.Tensor]:
        """Generate a dead leaves partition from the model.

        Returns:
            tuple[pd.DataFrame, torch.Tensor]: Dataframe of resulting leaves and their
                parameters, as well as the partition.
        """
        leaves_params = []
        partition = torch.zeros(self.size, device=self.device)
        leaf_idx = 1

        while torch.any((partition == 0) & (self.position_mask == 1)):
            params = self.sample_parameters()
            leaf_mask = self.generate_leaf_mask(params)
            mask = leaf_mask & (partition == 0)
            if (mask.sum() > 0) & self.position_mask[
                params["y_pos"].to(int), params["x_pos"].to(int)
            ]:
                partition[mask] = leaf_idx
                leaves_params.append(params)
                leaf_idx += 1
            if (self.n_sample is not None) and leaf_idx >= self.n_sample:
                break

        leaves = pd.DataFrame(leaves_params, columns=self.params)
        leaves["leaf_idx"] = torch.tensor(range(leaf_idx - 1)) + 1
        return leaves, partition

    def _circular_leaf_mask(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate mask of circle from given area and x-y-position on tensor.

        Args:
            params (dict[str, tensor]): Value for each parameter.

        Returns:
            torch.Tensor: Leaf mask.
        """
        dist_from_center = torch.sqrt(
            (self.X - params["x_pos"]) ** 2 + (self.Y - params["y_pos"]) ** 2
        )
        mask = dist_from_center <= torch.sqrt(params["area"] / torch.pi)
        return mask

    def _rectangular_leaf_mask(self, params: dict) -> torch.Tensor:
        """Generate mask of rectangle from given area, aspect ratio, orientation,
        and x-y-position on tensor.

        Args:
            params (dict[str, tensor]): Value for each parameter.

        Returns:
            torch.Tensor: Leaf mask.
        """
        height = torch.sqrt(params["area"] / params["aspect_ratio"])
        width = height * params["aspect_ratio"]
        sin = torch.sin(params["orientation"])
        cos = torch.cos(params["orientation"])
        dx = self.X - params["x_pos"]
        dy = self.Y - params["y_pos"]
        X = dx * cos - dy * sin
        Y = dx * sin + dy * cos
        mask = (torch.abs(X) <= width / 2) & (torch.abs(Y) <= height / 2)
        return mask

    def _ellipsoid_leaf_mask(self, params: dict) -> torch.Tensor:
        """Generate mask of ellipsoid from given area, aspect ratio, orientation,
        and x-y-position on tensor.

        Args:
            params (dict[str, tensor]): Value for each parameter.

        Returns:
            torch.Tensor: Leaf mask.
        """
        a = torch.sqrt((params["area"] * params["aspect_ratio"]) / torch.pi)
        b = torch.sqrt(params["area"] / (torch.pi * params["aspect_ratio"]))
        sin = torch.sin(params["orientation"])
        cos = torch.cos(params["orientation"])
        dx = self.X - params["x_pos"]
        dy = self.Y - params["y_pos"]
        X = dx * cos - dy * sin
        Y = dx * sin + dy * cos
        mask = (X / a) ** 2 + (Y / b) ** 2 <= 1
        return mask

    def _regular_polygon_leaf_mask(self, params: dict) -> torch.Tensor:
        """Generate mask of regular polygon from given area, number of vertices
        and x-y-position on tensor.

        Args:
            params (dict[str, tensor]): Value for each parameter.

        Returns:
            torch.Tensor: Leaf mask.
        """
        radius = torch.sqrt(
            2
            * params["area"]
            / (params["n_vertices"] * torch.sin(2 * torch.pi / params["n_vertices"]))
        )
        angles = torch.linspace(
            0,
            2 * torch.pi,
            params["n_vertices"].int(),
            device=self.device,
        )
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        vertices = torch.stack(
            (
                params["x_pos"] + radius * cos_angles,
                params["y_pos"] + radius * sin_angles,
            ),
            dim=1,
        )
        n = vertices.size(0)

        x_coords, y_coords = self.X.ravel(), self.Y.ravel()
        mask = torch.zeros(x_coords.shape[0], device=self.device, dtype=torch.bool)

        # ray casting algorithm
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            y_range_condition = (v1[1] > y_coords) != (v2[1] > y_coords)
            x_intersection = (v2[0] - v1[0]) * (y_coords - v1[1]) / (
                v2[1] - v1[1]
            ) + v1[0]
            x_range_condition = x_coords < x_intersection

            mask ^= y_range_condition & x_range_condition

        return mask.reshape(self.size)


class DeadLeavesImage:
    """Setup color and texture model for a dead leaves partition.

    Args:
        - leaves (DataFrame): Dataframe of leaves and their parameters.
        - partition (tensor): Partition of the image area.
        - color_param_distributions (dict[str, dict[str, dict[str, float]]]):
            Color parameters and their distribution setup.
        - texture_param_distributions (dict[str, dict[str, dict[str, float]]]):
            Texture parameters and their distribution setup.
        - background_color (tensor): For images which are not fully covered
            (due to a position mask or sparse sampling) one can set a RGB background color.
            If None the color and texture will be sampled from the distributions.
            Defaults to None.

        - device: Torch device to use, either cuda or cpu.
        - size (tuple[int,int]): Image size.
        - sample_colors (callable): Function to sample colors.
        - sample_texture (callable): Function to sample texture.
        - color_distributions (dict[str, Distribution]): Distribution of colors.
        - texture_distributions (dict[str, Distribution]): Distribution of texture.

    Methods:
        - model: Generate color_distributions and texture_distributions
            based on initialization.
        - sample_image: Sample color and texture for each leaf in the partition.
        - _sample_grayscale_colors: For each leaf sample a one-dimensional color.
        - _sample_RGB_colors: For each leaf sample a RGB color.
        - _sample_HSV_colors: For each leaf sample a HSV color.
        - _sample_colors_from_images: For each leaf sample a pixel color from a fixed
            but random image.
        - _sample_grayscale_texture: For each pixel sample a one-dimensional
            additive texture value.
        - _sample_RGB_texture: For each pixel sample a RGB additive texture value.
        - _sample_HSV_texture: For each pixel sample a HSV additive texture value.
        - show: Plot generated image.
        - save: Save image to directory.
        - animate: Generate animation of dead leaves sampling process.
    """

    def __init__(
        self,
        leaves: pd.DataFrame,
        partition: torch.Tensor,
        color_param_distributions: dict[str, dict[str, dict[str, float]]],
        texture_param_distributions: dict[str, dict[str, dict[str, float]]],
        background_color: torch.Tensor | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = partition.shape
        self.color_param_distributions = color_param_distributions
        self.texture_param_distributions = texture_param_distributions
        self.background_color = background_color
        self.leaves = leaves
        self.partition = partition
        colors = {
            frozenset(["R", "G", "B"]): self._sample_RGB_colors,
            frozenset(["H", "S", "V"]): self._sample_HSV_colors,
            frozenset({"gray"}): self._sample_grayscale_colors,
            frozenset({"source"}): self._sample_colors_from_images,
        }
        textures = {
            frozenset(["R", "G", "B"]): self._sample_RGB_texture,
            frozenset(["H", "S", "V"]): self._sample_HSV_texture,
            frozenset({"gray"}): self._sample_grayscale_texture,
        }
        self.sample_colors = colors[frozenset(self.color_param_distributions.keys())]
        self.sample_texture = textures[
            frozenset(self.texture_param_distributions.keys())
        ]
        self.model()

    def model(self) -> None:
        """Generate distribution instances to sample from"""
        with self.device:
            self.color_distributions = {}
            for param, dist_dict in self.color_param_distributions.items():
                dist_name = list(dist_dict.keys())[0]
                dist_class = dist_kw[dist_name]
                hyper_params_dict = dist_dict[dist_name]
                hyper_params = list(hyper_params_dict.values())
                self.color_distributions[param] = dist_class(*hyper_params)
            self.texture_distributions = {}
            for param, dist_dict in self.texture_param_distributions.items():
                dist_name = list(dist_dict.keys())[0]
                dist_class = dist_kw[dist_name]
                hyper_params_dict = dist_dict[dist_name]
                hyper_params = list(hyper_params_dict.values())
                self.texture_distributions[param] = dist_class(*hyper_params)

    def sample_image(self) -> torch.Tensor:
        """Generate a dead leaves image from the model.

        Returns:
            torch.Tensor: Dead leaves image tensor.
        """
        with self.device:
            image = torch.zeros(self.size + (3,), device=self.device)
            colors = self.sample_colors()
            texture = self.sample_texture()
            for leaf_idx in self.leaves.leaf_idx:
                image[self.partition == leaf_idx] = torch.clip(
                    colors[leaf_idx - 1] + texture[self.partition == leaf_idx], 0, 1
                )
            if self.background_color is not None:
                image[self.partition == 0] = self.background_color
            return image

    def _sample_grayscale_colors(self) -> torch.Tensor:
        """Sample a grayscale color for each leaf.

        Returns:
            torch.Tensor: Color values.
        """
        with self.device:
            for dist in self.color_distributions.values():
                colors = dist.sample((len(self.leaves), 1))
        return colors.expand(-1, 3)

    def _sample_RGB_colors(self) -> torch.Tensor:
        """Sample a RGB color for each leaf.

        Returns:
            torch.Tensor: Color values.
        """
        with self.device:
            colors = {}
            for param, dist in self.color_distributions.items():
                colors[param] = dist.sample((len(self.leaves),))
            return torch.stack(tuple(colors.values()), dim=1)

    def _sample_HSV_colors(self) -> torch.Tensor:
        """Sample a HSV color for each leaf.

        Returns:
            torch.Tensor: Color values.
        """
        with self.device:
            colors = {}
            for param, dist in self.color_distributions.items():
                colors[param] = dist.sample((len(self.leaves),))
            color_tensor = torch.stack(tuple(colors.values()), dim=1)
        return torch.Tensor(hsv_to_rgb(color_tensor.cpu())).to(self.device)

    def _sample_colors_from_images(self) -> torch.Tensor:
        """From a random image sample a pixel value for each leaf.

        Returns:
            torch.Tensor: Color values.
        """
        with self.device:
            for _, dist in self.color_distributions.items():
                image_path = dist.sample()
            image = pil_to_tensor(PIL.Image.open(image_path)) / 255
            image_vector = image.reshape((3, -1))
            idx = torch.multinomial(
                torch.ones(image_vector.shape[-1]), len(self.leaves), replacement=True
            )
            color_tensor = image_vector[:, idx]
            return color_tensor.permute(1, 0)

    def _sample_grayscale_texture(self) -> torch.Tensor:
        """Sample grayscale texture for each pixel.

        Returns:
            torch.Tensor: Texture values.
        """
        with self.device:
            for dist in self.texture_distributions.values():
                texture = dist.sample(self.size)
        return texture.repeat(3, 1, 1).permute(1, 2, 0)

    def _sample_RGB_texture(self) -> torch.Tensor:
        """Sample a RGB texture for each pixel.

        Returns:
            torch.Tensor: Texture values.
        """
        with self.device:
            texture = {}
            for param, dist in self.texture_distributions.items():
                texture[param] = dist.sample(self.size)
            return torch.stack(tuple(texture.values()), dim=2)

    def _sample_HSV_texture(self) -> torch.Tensor:
        """Sample a HSV texture for each pixel.

        Returns:
            torch.Tensor: Texture values.
        """
        with self.device:
            texture = {}
            for param, dist in self.texture_distributions.items():
                texture[param] = dist.sample(self.size)
            texture_tensor = torch.stack(tuple(texture.values()), dim=2)
        return torch.Tensor(hsv_to_rgb(texture_tensor.cpu())).to(self.device)

    def show(self, image: torch.Tensor) -> None:
        """Show selected image.

        Args:
            image (torch.Tensor): Image to show.
        """
        fig, ax = plt.subplots(frameon=False)
        ax.imshow(image.cpu().numpy(), vmax=1, vmin=0)
        fig.tight_layout()
        ax.axis("off")

        plt.show()

    def save(self, image: torch.Tensor, save_to: Path) -> None:
        """Save image to path.

        Args:
            image (torch.Tensor): Image to save.
            save_to (Path): Path to file to save image to.
        """
        plt.imsave(save_to, image.cpu().numpy())

    def animate(self, fps: int = 10, save_to: Path = None) -> animation.FuncAnimation:
        """Generate animation of dead leaves partition generation.

        Args:
            fps (int, optional): Frames per second of animation. In each frame a new
                leaf is sampled. Defaults to 10.
            save_to (Path, optional): Path to file to save animation to. If None the
                animation will not be saved. Defaults to None.

        Returns:
            animation.FuncAnimation: Animation of partition generation.
        """
        fig, ax = plt.subplots(frameon=False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(0, self.size[1])
        ax.set_ylim(0, self.size[0])

        def add_leaf(frame):
            leaf = self.leaves.iloc[frame]
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
            fig, add_leaf, frames=len(self.leaves), interval=1000 / fps, repeat=False
        )

        if save_to:
            FFwriter = animation.FFMpegWriter(fps=fps)
            dl_animation.save(save_to, writer=FFwriter)

        return dl_animation
