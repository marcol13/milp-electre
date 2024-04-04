class Node:
    def __init__(self, name, level=0):
        self.name = name
        self.level = level
        self.superiors = []

    def add_superior(self, node):
        self.superiors.append(node)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name