class Node:
    def __init__(self, name: str, level: int=0):
        self.name = name
        self.level = level
        self.superiors = []

    def add_superior(self, node: 'Node'):
        self.superiors.append(node)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name