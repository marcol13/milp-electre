from mcda.core.relations import Relation, PreferenceRelation, IncomparableRelation, IndifferenceRelation

class PositivePreference(PreferenceRelation):
    def __init__(self, a: any, b: any, name="P+"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        self.name = name

    def compatible(self, other: Relation):
        is_reflective = isinstance(other, NegativePreference) and self.a == other.b and self.b == other.a
        return self == other or is_reflective or not super().same_elements(other)


class NegativePreference(PreferenceRelation):
    def __init__(self, a: any, b: any, name="P-"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        self.name = name

    def compatible(self, other: Relation):
        is_reflective = isinstance(other, PositivePreference) and self.a == other.b and self.b == other.a
        return self == other or is_reflective or not super().same_elements(other)

class Incomparible(IncomparableRelation):
    def __init__(self, a: any, b: any, name="R"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        self.name = name

class Indifference(IndifferenceRelation):
    def __init__(self, a: any, b: any, name="I"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        self.name = name