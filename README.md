# hierarchical-classification

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/perellonieto/hierarchical_classification/HEAD)

You can run the Jupyter Notebooks online in Binder by clicking the **launch binder** button above.

# Installation

```
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Jupyter notebooks

After loading the virtual environment

```
source venv/bin/activate
```

Create a new Python kernel that can be later loaded from the Jupyter Notebook
environment. First ensure that you have installed the package ipykernel

```
pip install ipykernel
```

Then create a kernel with a given name

```
python -m ipykernel install --user --name hierarchical
```

Now it is possible to launch the Jupyter Notebook environment

```
jupyter notebook
```
