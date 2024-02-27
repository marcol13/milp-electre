import pandas as pd
from .relations import *
from .types import ScoreType, RelationType
from .const import RELATIONS

default_scoretable = {
    PositivePreference: {
        PositivePreference: 0,
        NegativePreference: 4,
        Indifference: 2,
        Incomparible: 3
    },
    NegativePreference: {
        PositivePreference: 4,
        NegativePreference: 0,
        Indifference: 2,
        Incomparible: 3
    },
    Indifference: {
        PositivePreference: 2,
        NegativePreference: 2,
        Indifference: 0,
        Incomparible: 2
    },
    Incomparible: {
        PositivePreference: 3,
        NegativePreference: 3,
        Indifference: 2,
        Incomparible: 0
    }
}

class Score:
    def __init__(self, score_matrix: ScoreType=default_scoretable):
        self.validate(score_matrix)
        self.score_matrix = score_matrix

    def show(self):
        df = pd.DataFrame(self.score_matrix)
        df.rename(columns=lambda x: x.name, index=lambda x: x.name, inplace=True)
        print(df)

    def get_distance(self, a:RelationType, b:RelationType):
        return self.score_matrix[a][b]

    def validate(self, score:ScoreType):
        if set(score.keys()) != set(RELATIONS):
            raise ValueError(f"There are incorrect keys. Dictionary should includes: {[rel.name for rel in RELATIONS]}")
        
        for a in score.keys():
            if set(score[a].keys()) != set(RELATIONS):
                raise ValueError(f"There are incorrect keys. Dictionary should includes classes for: {[rel.name for rel in RELATIONS]}")
            
            for b in score[a].keys():                
                if (score[a][b] > 4 or score[a][b] < 0):
                    raise ValueError("Score values should be between 0 and 4 included.")
                
    def print_status(self, prob):
        pass