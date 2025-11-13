This folder contains code to set up a dataset that sampels from the SCM. 

The core functionality is in Datasets/ObservationalDataset.py

An ObservationalDataset is a class that takes as input an SCM config, a Preprocessing config and a dataset config. 

- The SCM config determines hyperparameter distributions for the SCM. 
- The dataset config determines how the datasets are generated from the SCM, i.e. how many samples to draw. 
- The preprocessing config specifies standard preprocessing things for tabular data. 