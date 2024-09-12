# SIGMAP
This repository contains the code of the method described in "SIGMAP: an explainable artificial intelligence tool for SIGMA-1 receptor affinity Prediction" (add ref).

## Requisites

This package requires KNIME software and the following Python packages:
* rdkit
* IPython
* cairosvg
* torch
* enchant
* selfies

## Usage

`.\pipeline_plat.sh <input_file>.csv`

where `<input_file>.csv` contains the input SMILES formatted like the pb28.csv file in the example folder

The code will outputs the following files:

`<input_file>.csv_out`: contains the output of the prediction  \
`<input_file>.png`: is the SHAP analysis of the prediction     \
`<input_file>.csv_counter_out.csv`: contains the predition of the analogues molecules generated via DelaDrugSelf

