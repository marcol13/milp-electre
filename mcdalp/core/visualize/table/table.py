import numpy as np

from ...types import QuadraticArrayType
from abc import ABC, abstractmethod

class Table(ABC):
    def __init__(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType, labels: list[str], title: str = "", pos_color: str = "#00FF00", neg_color: str = "FF0000", **kwargs):
        self.src_data = src_data
        self.dest_data = dest_data
        self.labels = labels
        self.title = title
        self.pos_color = pos_color
        self.neg_color = neg_color
        self.kwargs = kwargs
        self.size = len(self.dest_data)
        self.mask = None
        self.rules = None

    @abstractmethod
    def get_mask(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType) -> QuadraticArrayType:
        pass

    def get_colors_matrix(self, mask: QuadraticArrayType, rules: dict) -> QuadraticArrayType:
        colors = np.zeros(mask.shape, dtype=str)
        for rule in rules.keys():
            colors = np.where(mask == rule, rules[rule], colors)
        colors = np.where(colors == "", "w", colors)
        return colors

    @abstractmethod
    def draw(self):
        pass
