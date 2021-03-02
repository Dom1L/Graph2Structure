# Graph2Structure

Graph2Structure (G2S) is a kernel ridge regression based machine learning model to
predict 3D atomic structures from graph based information (e.g. SMILES).

## Installation

Install all requirements using pip

```bash
pip install -r requirements.txt
```

To use G2S as a regular Python library, you have to add the repository to your `$PYTHONPATH`

```
export PYTHONPATH="/home/dominik/github/graph2structure"
```

Additional packages required to use features such as graph-extraction include

```python
rdkit
xyz2mol
xTB
```

## Distance Geometry

To reconstruct 3D coordinates from a distance matrix, you have to install a distance geometry solver such as DGSOL.

DGSOL is available at http://www.mcs.anl.gov/~more/dgsol/dgsol-1.3.tar.gz

Compile the software according to the README https://www.mcs.anl.gov/~more/dgsol/README and add 
the path to the dgsol binary to your `$PATH


## Usage

For usage examples, take a look at the tutorials in the example folder!


## References

```
@misc{2102.02806,
Author = {Dominik Lemm and Guido Falk von Rudorff and O. Anatole von Lilienfeld},
Title = {Energy-free machine learning predictions of {\em ab initio} structures},
Year = {2021},
Eprint = {arXiv:2102.02806},
}

```

