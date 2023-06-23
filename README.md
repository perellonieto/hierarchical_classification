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

There are some Python submodules in the folder **lib**. In order to install
them, first we need to initialise the submodules, and then update them.

```
git submodule init
git submodule update
```

(It is also possible to initiate and update all the submodules when clonning the
repository by using the command `git clone --recurse-submodules`.)

Install the hierarchical impurity library

```
pip install -e lib/himpurity/
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
