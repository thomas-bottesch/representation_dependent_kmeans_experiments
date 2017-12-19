# Reproducing Results

## General information
This repository's purpose is to make it easy to reproduce the results from the paper "Representation Dependent Distance Approximations for k-means"

Reproducing the results can be done in very few simple steps. All data sets that were used in the paper
are automatically downloaded when starting the scripts, make sure you have enough disk space. A linux environment is required!

## Installation of the required python packages
This should preferably be done within a python virtual environment.

First pip should be upgraded to the newest version:
```
pip install --upgrade pip
```

The required python packages need to be installed. Please do not install the requirements any other way since this
command ensures that the order of installation is the same as the ordering within the requirements file. 
```
cat requirements.txt | xargs -n 1 -L 1 pip install
```

## Running the experiments in testmode
Running the experiments can be very time consuming (weeks). To do it only on one (very small) data set you can use the test mode which finishes in a few minutes.
```
cd experiments 
python do_experiments.py --testmode
```

```
optional arguments:
  -h, --help            show this help message and exit
  --dataset_folder DATASET_FOLDER
                        Path datasets are downloaded to
  --output_path OUTPUT_PATH
                        Path to the results of single algorithm executions
  --output_path_latex OUTPUT_PATH_LATEX
                        Path to the results as latex tables
  --only_result_evaluation
                        Recreate latex tables based on previous results
                        without executing kmeans again
  --testmode            Only run the experiments for a single small dataset
```

## Running the experiments in full mode
If you want to do the full tests to create the full tables within the paper (can take multiple weeks)
```
cd experiments
python do_experiments.py
```

## Results for testmode & full mode
The results will be available in the respective folder:
```
cd experiments
output_path_latex/plot-memory-consumption-single.tex (This can directly be compiled)
output_path_latex/plot-memory-consumption.tex (This contains only the table and cannot be directly compiled)
output_path_latex/plot-speed-comparison-single.tex (This can directly be compiled)
output_path_latex/plot-speed-comparison.tex (This contains only the table and cannot be directly compiled)
```
