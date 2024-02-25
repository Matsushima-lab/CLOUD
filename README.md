# CLOUD

This is an implementation of the following papers:  
[1] Masatoshi Kobayashi, Kohei Miyaguchi, Shin Matsushima. (2024) [Detection of Unobserved Common Causes based on NML Code in Discrete, Mixed, and Continuous Variables](https://arxiv.org/abs/2403.06499). Preprint available on arXiv.<br>
[2] Masatoshi Kobayashi, Kohei Miyaguchi, Shin Matsushima, (2022) [Detection of Unobserved Common Cause in Discrete Data Based on the MDL Principle](https://ieeexplore.ieee.org/abstract/document/10020351/). IEEE BigData 2022.

## Requirement
- Python 3.8+
- [Rye](https://rye-up.com/) or pip

## Setup
### With Rye (Recommended)
After installing [Rye](https://rye-up.com/guide/installation/), executing the following command(s):
```bash
$ rye sync
$ rye shell # optional
```

You can now import and use the ```CLOUD``` class:
```python
from cloud import CLOUD
```

### Without Rye
Create a new virtual environment and install the dependencies:
```setup
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.lock
```



## Run CLOUD
### Sample Data
```bash
$ python -m unittest tests/test_cloud.py
```
If you are using Rye, you can also run the tests with:
```bash
$ rye run python -m unittest tests/test_cloud.py
```
