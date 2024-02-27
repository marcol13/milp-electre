from .outranking import Outranking

class CrispOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)
        self.variables = self.create_variable_matrix("outranking")
        self.solve()

    def solve(self, mode="complete"):
        pass