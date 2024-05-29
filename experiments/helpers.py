from mcdalp.core.relations import PositivePreference, NegativePreference, Indifference, Incomparible

score = {
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

def get_relation(a, b):
    if a > b:
        return PositivePreference
    elif a < b:
        return NegativePreference
    elif a == b == 1:
        return Indifference
    else:
        return Incomparible
    
def get_relation_score(rel_a, rel_b, score=score):
    return score[rel_a][rel_b]