# Nomad: Combined Geometric and Semantic Anomaly Detector

## Project Overview

**Nomad** is a system designed for **unsupervised anomaly detection** that combines two powerful methods: **local density/distance** and **cluster purity (semantic entropy)**. This dual approach ensures robustness against different types of outliers: those that are geographically isolated in the feature space (geometric anomalies) and those that are statistically mixed with other classes within a dense region (semantic anomalies). The goal was to have a simple pipeline that wasn't too sensitive to hyperparameters.

The system uses dimensionality reduction (currently **PCA**, hopefully **Autoencoders** soon) followed by clustering (**$K$-Means**, that will be replaced with **HDBSCAN**) and finally calculates a weighted anomaly score.

## Demo

The `Nomad` class is designed for a standard `fit`/`detect` pipeline.

```python
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

>>> [0.4932361  0.77853699 1.22404504 0.71898946 0.52439317]
```

![Demo images](https://github.com/TQCB/nomad/blob/main/assets/demo_image.png)