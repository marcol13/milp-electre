from mcda.core.relations import Relation, PreferenceRelation, IncomparableRelation, IndifferenceRelation

class PositivePreference(PreferenceRelation):
    name = "P+"
    def __init__(self, a: any, b: any, name="P+"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if PositivePreference.name != name:
            PositivePreference.name = name

    def compatible(self, other: Relation):
        is_reflective = isinstance(other, NegativePreference) and self.a == other.b and self.b == other.a
        return self == other or is_reflective or not super().same_elements(other)


class NegativePreference(PreferenceRelation):
    name = "P-"
    def __init__(self, a: any, b: any, name="P-"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if NegativePreference.name != name:
            NegativePreference.name = name

    def compatible(self, other: Relation):
        is_reflective = isinstance(other, PositivePreference) and self.a == other.b and self.b == other.a
        return self == other or is_reflective or not super().same_elements(other)

class Incomparible(IncomparableRelation):
    name = "R"
    def __init__(self, a: any, b: any, name="R"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if Incomparible.name != name:
            Incomparible.name = name

class Indifference(IndifferenceRelation):
    name = "I"
    def __init__(self, a: any, b: any, name="I"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if Indifference.name != name:
            Indifference.name = name