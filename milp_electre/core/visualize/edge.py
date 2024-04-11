from .node import Node

class Edge:
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b
    
    def __str__(self):
        return f"{self.a} -> {self.b}"
    
    def __repr__(self):
        return f"{self.a} -> {self.b}"
    
    def __eq__(self, other):
        return self.a == other.a and self.b == other.b
    
    def is_self_loop(self):
        return self.a == self.b
    
