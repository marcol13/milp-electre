import numpy as np

from matplotlib import pyplot as plt
from ..types import QuadraticArrayType
from typing import Literal
from ..relations import PositivePreference, NegativePreference, Indifference, Incomparible
from enum import Enum
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

    def get_colors_matrix(self, mask: QuadraticArrayType, rules: dict) -> QuadraticArrayType:
        colors = np.zeros(mask.shape, dtype=str)
        for rule in rules.keys():
            colors = np.where(mask == rule, rules[rule], colors)
        colors = np.where(colors == "", "w", colors)
        return colors

    @abstractmethod
    def draw(self):
        pass
        
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

class ValuedTable(Table):
    def __init__(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType, labels: list[str], title: str = "", pos_color: str = "#00FF00", neg_color: str = "#FF0000", **kwargs):
        super().__init__(src_data, dest_data, labels, title, pos_color, neg_color, **kwargs)
        self.mask = self.get_mask(self.src_data, self.dest_data)
        self.rules = {
            1: self.pos_color,
            0: self.neg_color,
            -1: 'w'
        }

    def get_mask(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType) -> QuadraticArrayType:
        mask = np.where(dest_data == 1, 1, 0)
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

class StochasticTable(Table):
    class RelationEnum(Enum):
        PositivePreference = 0
        NegativePreference = 1
        Indifference = 2
        Incomparible = 3

    def __init__(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType, labels: list[str], title: str = "", pos_color: str = "#00FF00", neg_color: str = "#FF0000", **kwargs):
        super().__init__(src_data, dest_data, labels, title, pos_color, neg_color, **kwargs)
        self.dest_data = self.dest_data.astype(int)
        self.src_mask = self.get_mask(self.src_data, self.dest_data, "src")
        self.dest_mask = self.get_mask(self.src_data, self.dest_data, "dest")
        self.rules = {
            1: self.pos_color,
            0: 'w',
            -1: 'w'
        }

    def get_mask(self, src_data: QuadraticArrayType, dest_data: QuadraticArrayType, type: Literal["src", "dest"]) -> QuadraticArrayType:
        if type == "src":
            mask = np.zeros((4, self.size, self.size), dtype=int)
            for i in range(len(dest_data)):
                for j in range(len(dest_data)):
                    if i != j:
                        val = dest_data[i][j]
                        rev_val = dest_data[j][i]
                        matrix_index = self.__get_relation(val, rev_val)
                        mask[matrix_index][i][j] = 1
            return mask
        elif type == "dest":
            mask = dest_data.copy()
            np.fill_diagonal(mask, -1)
            return mask
        else:
            raise ValueError(f"Type {type} is not supported.")

    def __get_relation(self, i: int, j: int):
        if (i,j) == (1,0):
            return self.RelationEnum.PositivePreference.value
        elif (i,j) == (0,1):
            return self.RelationEnum.NegativePreference.value
        elif (i,j) == (1,1):
            return self.RelationEnum.Indifference.value
        else:
            return self.RelationEnum.Incomparible.value
    
    def draw(self):
        params = {
            'rowLabels': self.labels,
            'colLabels': self.labels,
            'loc': 'center',
            'rowLoc': 'center',
            'cellLoc': 'center',
            **self.kwargs
        }

        plt.tight_layout()
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        colors = self.get_colors_matrix(self.dest_mask, self.rules)
        self.__prepare_table(ax1, params, "Output Matrix", self.dest_data, colors)

        for idx, rel_matrix in enumerate(self.src_data.values()):
            ax = plt.subplot2grid((3, 2), (1 + idx // 2, idx % 2))
            rel_name = list(self.src_data.keys())[idx]
            colors = self.get_colors_matrix(self.src_mask[idx], self.rules)
            self.__prepare_table(ax, params, f"{rel_name.full_name} Matrix", rel_matrix, colors)
        plt.show()

    def __prepare_table(self, subplot, params: dict, title: str, cellText, cellColours):
        subplot.axis('tight')
        subplot.axis('off')

        params['cellText'] = cellText
        params['cellColours'] = cellColours
        subplot.set_title(title, fontweight='bold', fontsize=15)
        table = subplot.table(**params)
        table._autoColumns = []

        for cell in table.get_celld().values():
            cell.set_width(.125)
            cell.set_height(.15)

def table_factory(table_type: str, **args):
    if table_type == "crisp":
        return CrispTable(**args)
    else:
        raise ValueError(f"Table type {table_type} is not supported.")