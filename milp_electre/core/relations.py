from mcda.core.relations import Relation, PreferenceRelation, IncomparableRelation, IndifferenceRelation

class PositivePreference(PreferenceRelation):
    short_name = "P+"
    full_name = "Positive Preference"
    def __init__(self, a: any, b: any, short_name="P+"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if PositivePreference.short_name != short_name:
            PositivePreference.short_name = short_name

    def compatible(self, other: Relation):
        is_reflective = isinstance(other, NegativePreference) and self.a == other.b and self.b == other.a
        return self == other or is_reflective or not super().same_elements(other)
    
    def __str__(self):
        return PositivePreference.full_name


class NegativePreference(PreferenceRelation):
    short_name = "P-"
    full_name = "Negative Preference"
    def __init__(self, a: any, b: any, short_name="P-"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if NegativePreference.short_name != short_name:
            NegativePreference.short_name = short_name

    def compatible(self, other: Relation):
        is_reflective = isinstance(other, PositivePreference) and self.a == other.b and self.b == other.a
        return self == other or is_reflective or not super().same_elements(other)
    
    def __str__(self):
        return NegativePreference.full_name

class Incomparible(IncomparableRelation):
    short_name = "R"
    full_name = "Incomparible"
    def __init__(self, a: any, b: any, short_name="R"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if Incomparible.short_name != short_name:
            Incomparible.short_name = short_name

    def __str__(self):
        return Incomparible.full_name

class Indifference(IndifferenceRelation):
    short_name = "I"
    full_name = "Indifference"
    def __init__(self, a: any, b: any, short_name="I"):
        super().__init__(a, b)
        self.a = a
        self.b = b
        if Indifference.short_name != short_name:
            Indifference.short_name = short_name

    def __str__(self):
        return Indifference.full_name