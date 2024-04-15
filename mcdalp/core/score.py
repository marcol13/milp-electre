import pandas as pd
from .relations import *
from .types import ScoreType, RelationType
from .const import RELATIONS, DEFAULT_SCORETABLE
from .utils import check_monotonicity, check_keys

class Score:
    def __init__(self, score_matrix: ScoreType=DEFAULT_SCORETABLE):
        self.validate(score_matrix)
        self.score_matrix = score_matrix

    def show(self):
        df = pd.DataFrame(self.score_matrix)
        df.rename(columns=lambda x: x.short_name, index=lambda x: x.short_name, inplace=True)
        print(df)

    def get_distance(self, a:RelationType, b:RelationType):
        return self.score_matrix[a][b]
    
    def validate(self, score:ScoreType):
        check_keys(score, RELATIONS)
        
        for a in score.keys():
            check_keys(score[a], RELATIONS)
            for b in score[a].keys():
                if a == b and score[a][b] != 0:
                    raise ValueError("Self relation must be equal to 0")              
                elif (score[a][b] != score[b][a]):
                    raise ValueError("Score values for the same relations are different.")
                
        check_monotonicity(score, DEFAULT_SCORETABLE)
