import numpy as np

from matplotlib import pyplot as plt
from ..types import QuadraticArrayType
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

    @abstractmethod
    def get_mask(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType) -> QuadraticArrayType:
        pass

    @abstractmethod
    def get_colors_matrix(self, mask: QuadraticArrayType) -> QuadraticArrayType:
        pass

    @abstractmethod
    def draw(self):
        pass
        
class CrispTable(Table):
    def __init__(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType, labels: list[str], title: str = "", pos_color: str = "#00FF00", neg_color: str = "#FF0000", **kwargs):
        super().__init__(src_data, dest_data, labels, title, pos_color, neg_color, **kwargs)
        self.mask = self.get_mask(self.src_data, self.dest_data)

    def get_mask(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType) -> QuadraticArrayType:
        mask = np.where(src_data == dest_data, 1, 0)
        np.fill_diagonal(mask, -1)
        mask = mask.astype(int)
        
        return mask
    
    def get_colors_matrix(self, mask: QuadraticArrayType) -> QuadraticArrayType:
        colors = np.zeros(mask.shape, dtype=str)
        colors = np.where(mask == 1, self.pos_color, colors)  # Swap pos_color and neg_color
        colors = np.where(mask == 0, self.neg_color, colors)  # Swap pos_color and neg_color
        colors = np.where(mask == -1, 'w', colors)  # Swap pos_color and neg_color
        return colors

    def draw(self):
        colors = self.get_colors_matrix(self.mask)
        _, axs = plt.subplots(2)

        params = {
            'cellText': self.src_data,
            'cellColours': colors,
            'rowLabels': self.labels,
            'colLabels': self.labels,
            'loc': 'center',
            'rowLoc': 'center',
            'cellLoc': 'center',
            **self.kwargs
        }

        plt.tight_layout()
        self.__prepare_table(axs[0], params, "Input Matrix")

        params['cellText'] = self.dest_data
        self.__prepare_table(axs[1], params, "Output Matrix")
        plt.show()

    def __prepare_table(self, subplot, params: dict, title: str):
        subplot.axis('tight')
        subplot.axis('off')

        subplot.set_title(title, fontweight='bold', fontsize=15)
        table = subplot.table(**params)
        table._autoColumns = []

        for cell in table.get_celld().values():
            cell.set_width(.1)
            cell.set_height(.1)

def table_factory(table_type: str, **args):
    if table_type == "crisp":
        return CrispTable(**args)
    else:
        raise ValueError(f"Table type {table_type} is not supported.")