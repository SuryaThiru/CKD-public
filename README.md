# Prediction Models for Chronic Kidney Disease using Taiwan NHI data

The repository contains code and model exports to reproduce the results on the paper titled "Machine Learning Prediction Models for Chronic Kidney Disease using National Health Insurance Claim Data in Taiwan".

The dataset and models can be obtained from this link: https://osf.io/j3gur/

Set up a python 3.7 virtual environment and install the dependencies:

```
pip install -r requirements.txt
```

### Note
* The scripts currently only work on linux, because of hardcoded file paths.
* GPU and 16G RAM is recommended.


## Dataset

Extract the data in the `data.zip` archive into a `data` directory in the project root.

age_sex.feather: feather file containing the patient IDs, age, sex and the target variable

test_ids.pkl: patient IDs that are part of the test set

top_100_features_mask.joblib: dump of boolean mask of the top 100 features to filter from the 2D data

flat.feather: aggregated data

dict.pkl: pickle of python dictionary mapping patient ID to scipy csr matrix of 2D diag/drug data. This format was easy to handle large number of highly sparse 2D matrices.

## Models

Extract the models in the `models.zip` archive into a `models` directory in the project root.

The sklearn models are saved as joblib dumps, and the tf models are saved as hdf5 files. Ensure the appropriate versions are used for the dependencies to prevent conflicts during de-serialization.

### Abbreviations used

agg: aggregate models \
TS: Time Series \
ST: temporal-quarterly (Semi-Temporal)

## Code

preprocessing.py: Simple preprocessing for the aggregated models

data_loader.py: The custom data loader utility for keras/tf2

results.ipynb: jupyter notebook to evaluate the models and generate the metrics and plots

Once the data and models are placed in the project root, run the `results.ipynb` notebook from jupyter.
