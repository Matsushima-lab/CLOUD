# CLOUD

This repository implements CLOUD for bivariate causal discovery when unobserved common causes may exist.

Given paired observations of two variables, CLOUD compares four causal explanations: 

- `X -> Y`: `X` causes `Y`
- `X <- Y`: `Y` causes `X`
- `X ⫫ Y`: `X` and `Y` are independent
- `X <- C -> Y`: unobserved common causes influence both `X` and `Y`

CLOUD selects the explanation with the shortest codelength based on the MDL principle. The method supports discrete, mixed, and continuous variables.


## Usage

```python
from cloud import CLOUD

X = [0, 0, 1, 1, 2, 2, 2, 1]
Y = [1, 1, 2, 2, 0, 0, 0, 2]

model = CLOUD(
    X,
    Y,
    is_X_continuous=False,
    is_Y_continuous=False,
).fit()

print(model.predict())
print(model.summary())
```
`predict()` returns the inferred causal graph. `summary()` returns a formatted string with the codelength of each candidate model.

For continuous or mixed data, set `is_X_continuous` and `is_Y_continuous` accordingly.

## Citation
If you use this repository in your research, please cite the journal version.

```bibtex
@article{kobayashi2026detection,
  title={Detection of unobserved common causes under additive noise models based on NML code for discrete, mixed, and continuous variables},
  author={Kobayashi, Masatoshi and Miyaguchi, Kohei and Matsushima, Shin},
  journal={Data Mining and Knowledge Discovery},
  volume={40},
  number={5},
  pages={68},
  year={2026},
  publisher={Springer},
  doi={10.1007/s10618-025-01166-8},
  url={https://doi.org/10.1007/s10618-025-01166-8}
}
```

The original discrete-data version was presented at IEEE Big Data 2022.
```bibtex
@inproceedings{kobayashi2022detection,
  title={Detection of Unobserved Common Cause in Discrete Data Based on the MDL Principle},
  author={Kobayashi, Masatoshi and Miyaguchi, Kohei and Matsushima, Shin},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={45--54},
  year={2022},
  organization={IEEE}
}
```