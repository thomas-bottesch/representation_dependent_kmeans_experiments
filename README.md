# Reproducing Results

## General information
This repository's purpose is to make it easy to reproduce the results from the paper "Representation Dependent Distance Approximations for k-means"

Reproducing the results can be done in very few simple steps. All data sets that were used in the paper
are automatically downloaded when starting the scripts, make sure you have enough disk space. A linux environment is required!

## Installation of the required python packages
This should preferably done within a python virtual environment
First pip should be upgraded to the newest version
```
pip install --upgrade pip
```

Then the required python packages need to be installed. Please do not install the requirements any other way since this
command ensures that the order of installation is the same as the ordering within the requirements file. 
```
cat requirements.txt | xargs -n 1 -L 1 pip install
```

## Running the experiments in testmode
Running the experiments can be very time consuming. To do it only on one (very small) data set you can use the test mode.
cd memory_consumption 
python memory_consumption.py --testmode

cd speed_comparison
python dataset_speed_comparison.py --testmode

## Running the experiments in full mode
If you want to do the full tests to create the full tables within the paper (can take multiple weeks)
cd memory_consumption
python memory_consumption.py

cd speed_comparison
python dataset_speed_comparison.py

## Avoid duplicate executions
It is not needed to run the algorithms twice so if e.g. dataset_speed_comparison was already finished, the results for
memory_consumption.py can be retrieved directly and vice versa by doing:

cd memory_consumption
python memory_consumption.py --only_result_evaluation --output_path ../speed_comparison/output_path

cd speed_comparison
python dataset_speed_comparison.py --only_result_evaluation --output_path ../memory_consumption/output_path

## Results for testmode & full mode
The results will be available in the respective folder:

memory_consumption/output_path_csv/plot-memory-consumption-single.tex (This can directly be compiled)
memory_consumption/output_path_csv/plot-memory-consumption.tex (This contains only the table and cannot be directly compiled)
speed_comparison/output_path_csv/plot-speed-comparison-single.tex (This can directly be compiled)
speed_comparison/output_path_csv/plot-speed-comparison.tex (This contains only the table and cannot be directly compiled)
