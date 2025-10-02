from nomad import Nomad

import numpy as np
from sklearn.datasets import load_digits

raw_data = load_digits()
x = raw_data.data
y = raw_data.target

nomad = Nomad()
nomad.fit(x)

anomaly_scores = nomad.detect(x, y)

print(anomaly_scores[:5])