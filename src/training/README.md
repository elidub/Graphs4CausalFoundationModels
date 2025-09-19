This folder contains the code to actually train a model. 

In simple_run.py the actual main runner file is defined. It loads a config file, sets up logging, initializes the dataset, dataloader, bar-distribution, torch model and the trainer. 

Please note that to avoid issues with multiprocessing (on some servers) the dataloader is initialized very early in simple_run.py. 

The trainer is implemented in simplepfn_trainer.py.

There are three potentially very useful checks implemented in the checks folder:

- inspect_dataloader_samples.py: Visualizes samples from the dataloader together with a bunch of statistics. You might want to change the CONFIG_PATH argument to point to your config file. The results are saved in the checks/ResultsDataloaderSamples folder in a specific run folder. 

- baseline_performance.py: Checks how well simple ML models do on samples from the dataloader. You might also want to change the CONFIG_PATH argument to point to your config file. The results are saved in the checks/ResultsBaselines folder in a specific run folder.

- test_simple_run.py: Runs the entire trainig pipeline for a few iterations on CPU to check that everything works. You might want to change the CONFIG_PATH argument to point to your config file. You might want to carefully check the output on the console to see if training worked, i.e. the loss decreases a little bit.