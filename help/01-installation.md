# Installation

This example uses [mamba](https://github.com/conda-forge/miniforge), but conda will work just fine.  

Steps 2 and 3 give you more fine control over which CUDA and matching pytorch version to use. 

1\. Create a clean virtual environment 
```
mamba create -n bioencoder python=3.9
mamba activate bioencoder
```

2\. Install CUDA - e.g.:
```
mamba install cuda-toolkit==12.1*
```

3\. Install pytorch (check https://pytorch.org/get-started/locally/) - e.g.:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4\. Install bioencoder from pypi:
````
pip install bioencoder
````
