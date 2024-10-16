
# How to use this:

This repo is originated from: https://github.com/Bondify/gtfs_functions, but needed some modifications in order to handle the mav data.
 [Original readme](README_orig.md) 
## Create environment
- creating the conda environment described by ```environment.yml```:
```conda env create --file environment.yml``` 
- using the gtfs_functions package in development mode (that is, the changes in the packages will appear immediately, when we try to use it in ```ipynb```)
go to source folder  -> ``` conda develop .``` 
 
- adding the conda environment to jupyter server:
``` conda install ipykernel```

``` ipython kernel install --user --name=gtfs_venv```

After these, you can use ```gtfs_venv``` in jupyter 

## Example usage of the ```gtfs_functions``` package :
```mav.ipynb```

## Change codes:
the functions are installed in a development mode, so feel free to change them/add to them.

## keplergl 


[keplergl](https://kepler.gl/) is a quite cool visualisation tool, in theory it can be used to display things in a notebook, however i could not make it work for myself.
In this case as in ```mav.ipynb``` you can save to html, and open with browser (not sure if the export )

You can try the following things for yourself:

https://github.com/keplergl/kepler.gl/issues/583

https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html

https://stackoverflow.com/questions/76893872/modulenotfounderror-no-module-named-notebook-base-when-installing-nbextension


Some of these commands might be helpful:

```conda install ipywidgets```

```conda install -c conda-forge keplergl``` 

``` pip install jupyter_contrib_nbextensions```

```pip install --upgrade notebook==6.4.12```

```jupyter contrib nbextension install --user```

```pip install --upgrade jupyterthemes```

```jupyter nbextension install --py --sys-prefix keplergl```

```jupyter nbextension enable --py --sys-prefix keplergl```
