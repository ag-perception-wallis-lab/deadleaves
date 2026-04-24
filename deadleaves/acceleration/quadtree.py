"""
Spatial acceleration for dead leaves coverage tracking.

A quadtree that tracks which regions of the canvas still contain uncovered
pixels.  Fully covered subtrees are pruned from future queries, so late in
the sampling process, when most of the canvas is filled, only the sparse
frontier of remaining gaps is visited.
"""

from __future__ import annotations
import torch

__all__ = ["CoverageQuadTree"]


class Node:
    """Single node of the coverage quadtree.

    Leaf nodes (at the finest subdivision level) directly track whether their
    tile region contains any uncovered pixel.  Internal nodes summarise the
    status of their four children.

    Attributes:
        y0, x0, y1, x1:  half-open pixel region [y0, y1) x [x0, x1).
        alive:           True while at least one uncovered pixel exists in
                         this node's region.
        children:        None for leaf nodes, else a list of four Nodes.
    """

    __slots__ = ("y0", "x0", "y1", "x1", "alive", "children")

    def __init__(self, y0: int, x0: int, y1: int, x1: int) -> None:
        self.y0 = y0
        self.x0 = x0
        self.y1 = y1
        self.x1 = x1
        self.alive = True
        self.children: list[Node] | None = None

    # -- convenience -----------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def _slice(self) -> tuple[slice, slice]:
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def overlaps(self, y_min: int, x_min: int, y_max: int, x_max: int) -> bool:
        return (
            self.x0 < x_max and x_min < self.x1 and self.y0 < y_max and y_min < self.y1
        )


class CoverageQuadTree:
    """Quadtree that tracks uncovered pixels in the segmentation map.

    Build once before the sampling loop, then call :meth:`query_live_tiles`
    to get only the leaf tiles that (a) still contain uncovered pixels *and*
    (b) overlap a given AABB.  After writing into the segmentation map,
    call :meth:`update_tile` on each affected tile to propagate coverage
    status upward.

    Args:
        image_shape:
            (H, W) of the canvas.
        position_mask:
            Integer tensor (H, W) — 1 where sampling is allowed.  Tiles
            that fall entirely outside the position mask are born dead.
        min_tile:
            Minimum tile edge length.  Must be a power of two.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        position_mask: torch.Tensor,
        segmentation_map: torch.Tensor,
        min_tile: int = 16,
    ) -> None:
        H, W = image_shape
        self._seg: torch.Tensor = segmentation_map
        "Segmentation map"
        self._pos: torch.Tensor = position_mask
        "Position mask"
        self._min_tile: int = min_tile
        "Minimum tile edge length."

        # Pad conceptual extent to a power-of-two square so the recursive
        # split is clean. Nodes that overhang the actual canvas are born
        # dead.
        extent = 1
        while extent < max(H, W):
            extent *= 2
        self._H: int = H
        "Height of canvas"
        self._W: int = W
        "Width of canvas"

        self._root: Node = self._build(0, 0, extent, extent)
        "Root node of Quadtree"

    # -- construction ----------------------------------------------------------

    def _build(self, y0: int, x0: int, y1: int, x1: int) -> Node:
        """Build node for given canvas region

        Args:
            y0, x0, y1, x1:
                Half-open pixel region [y0, y1) x [x0, x1).
        """
        node = Node(y0, x0, y1, x1)

        # Completely outside actual canvas → dead
        if y0 >= self._H or x0 >= self._W:
            node.alive = False
            return node

        # Clip to canvas for mask checks
        cy1 = min(y1, self._H)
        cx1 = min(x1, self._W)

        # Check whether there are *any* uncovered, mask-allowed pixels in
        # this region.
        region_mask = self._pos[y0:cy1, x0:cx1]
        region_seg = self._seg[y0:cy1, x0:cx1]
        has_uncovered = torch.any((region_seg == 0) & (region_mask == 1)).item()
        if not has_uncovered:
            node.alive = False
            return node

        # Subdivide if the tile is large enough
        h = y1 - y0
        w = x1 - x0
        if h > self._min_tile and w > self._min_tile:
            my = y0 + h // 2
            mx = x0 + w // 2
            node.children = [
                self._build(y0, x0, my, mx),
                self._build(y0, mx, my, x1),
                self._build(my, x0, y1, mx),
                self._build(my, mx, y1, x1),
            ]
            # If all children are dead after build, collapse
            if not any(c.alive for c in node.children):
                node.alive = False

        return node

    @property
    def has_live_nodes(self) -> bool:
        """True while any uncovered, mask-allowed pixel remains."""
        return self._root.alive

    def query_live_tiles(
        self,
        y_min: int,
        x_min: int,
        y_max: int,
        x_max: int,
    ) -> list[Node]:
        """Return all alive leaf nodes overlapping the given AABB.

        Args:
            y_min, x_min, y_max, x_max:
                Query bounding box (not clipped — can exceed canvas).

        Returns:
            List of alive leaf Node objects whose regions intersect the
            query box.
        """
        result: list[Node] = []
        self._query(self._root, y_min, x_min, y_max, x_max, result)
        return result

    def _query(
        self,
        node: Node,
        y_min: int,
        x_min: int,
        y_max: int,
        x_max: int,
        out: list[Node],
    ) -> None:
        """Recursively collect all leaf nodes that overlap a query rectangle.

        Starting from ``node``, this method traverses the tree and appends
        all *alive* leaf nodes whose bounding boxes overlap the given
        axis-aligned query region to ``out``.

        Args:
            node (Node):
                The current node to test and recurse into.
            y_min, x_min, y_max, x_max:
                Boundaries of query rectangle.
            out (list[Node]):
                List that is populated in-place with all matching leaf nodes.
        """
        if not node.alive:
            return
        if not node.overlaps(y_min, x_min, y_max, x_max):
            return
        if node.is_leaf:
            out.append(node)
            return
        for child in node.children:  # pyright: ignore[reportOptionalIterable]
            self._query(child, y_min, x_min, y_max, x_max, out)

    def update_tile(self, tile: Node) -> None:
        """Re-check a leaf tile and propagate status up the tree.

        Call this after writing pixels into the segmentation map within
        *tile*'s region.  If the tile is now fully covered, it is marked
        dead and the change bubbles up.
        """
        # Clip to actual canvas
        cy1 = min(tile.y1, self._H)
        cx1 = min(tile.x1, self._W)
        if tile.y0 >= cy1 or tile.x0 >= cx1:
            tile.alive = False
            self._propagate_up()
            return

        region_seg = self._seg[tile.y0 : cy1, tile.x0 : cx1]
        region_mask = self._pos[tile.y0 : cy1, tile.x0 : cx1]
        still_uncovered = torch.any((region_seg == 0) & (region_mask == 1)).item()

        if not still_uncovered:
            tile.alive = False
            self._propagate_up()

    def _propagate_up(self) -> None:
        """Walk the entire tree bottom-up and collapse dead subtrees.

        This is cheap because the tree has O(N / min_tile²) leaf nodes and
        is only called when a tile dies — which can happen at most once per
        tile over the whole run.
        """
        self._propagate_node(self._root)

    def _propagate_node(self, node: Node) -> bool:
        """Returns True if node is alive."""
        if not node.alive:
            return False
        if node.is_leaf:
            return node.alive
        alive = False
        for child in node.children:  # pyright: ignore[reportOptionalIterable]
            if self._propagate_node(child):
                alive = True
        node.alive = alive
        return alive
