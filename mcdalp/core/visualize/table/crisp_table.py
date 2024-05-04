import numpy as np

from matplotlib import pyplot as plt
from .table import Table
from ...types import QuadraticArrayType

class CrispTable(Table):
    def __init__(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType, labels: list[str], title: str = "", pos_color: str = "#00FF00", neg_color: str = "#FF0000", **kwargs):
        super().__init__(src_data, dest_data, labels, title, pos_color, neg_color, **kwargs)
        self.mask = self.get_mask(self.src_data, self.dest_data)
        self.src_data = self.src_data.astype(np.int8)
        self.dest_data = self.dest_data.astype(np.int8)
        self.rules = {
            1: self.pos_color,
            0: self.neg_color,
            -1: 'w'
        }

    def get_mask(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType) -> QuadraticArrayType:
        mask = np.where(src_data == dest_data, 1, 0)
        np.fill_diagonal(mask, -1)
        mask = mask.astype(int)
        
        return mask

    def draw(self):
        colors = self.get_colors_matrix(self.mask, self.rules)
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