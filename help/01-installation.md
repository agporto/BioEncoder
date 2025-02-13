# Installation

This example uses [mamba](https://github.com/conda-forge/miniforge), but conda will work just fine.  

Steps 2 and 3 give you more fine control over which CUDA and matching pytorch version to use. 

1\. Create a clean virtual environment 
```
mamba create -n bioencoder python=3.11 #we tested BioeEcoder with Python 3.9 - 3.11
mamba activate bioencoder
```

2\. Install the CUDA toolkit (just an example, you can use different versions):
```
mamba install cuda-toolkit==12.6.3
```

3\. Install pytorch (needs to match your CUDA toolkit version - check https://pytorch.org/get-started/locally/) - e.g.:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

4\. Install bioencoder:
````
pip install bioencoder
````
