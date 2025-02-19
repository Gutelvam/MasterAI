from enum import Enum

class LearningType(Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    APRIORI = "apriori"

class DataClassification(Enum):
	NUMERICAL = "numerical"
	CATEGORICAL = "categorical"

class MissingValuesStrategy(Enum):
	MEAN="mean"
	MEDIAN = "median"
	MODE = "mode"
	CONSTANT = "constant"




