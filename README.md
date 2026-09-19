# Indian Monsoon EPCP Classification Using CNNs

This repository contains the convolutional neural network (CNN), data-processing,
and plotting scripts associated with the manuscript:

> *On the application of a Convolutional Neural Network to Identify Extreme
> Southwest Monsoon Circulation Patterns over Homogeneous Precipitation Zones
> of India*

The analysis identifies circulation patterns associated with extreme
precipitation days across six India Meteorological Department homogeneous
rainfall regions using ERA5 and NCEP-NCAR Reanalysis 1 fields.

## Repository structure

- `training_ERA5/`: ERA5 CNN training and supporting utilities.
- `training_R1/`: NCEP-NCAR Reanalysis 1 preprocessing, CNN training, and
  supporting utilities.
- `plotting_codes/`: scripts used to produce the main and supporting figures.

## Data

Large input datasets and intermediate products are not included. The scripts
require local copies of:

- IMD 0.25-degree daily gridded rainfall;
- ERA5 pressure-level and single-level reanalysis fields;
- NCEP-NCAR Reanalysis 1 fields; and
- the boundary files used for the IMD homogeneous rainfall regions.

Data paths are supplied through the `EPCP_DATA_DIR` environment variable in the
plotting scripts. Some training utilities retain the directory layout used for
the original analysis; update those paths to match the location of the input
data on the system where the analysis is run.

## Environment

Create a Python environment and install the listed packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cartopy and its GEOS/PROJ dependencies may be easier to install from
conda-forge.

## Reproducibility note

Random seeds used by the training scripts are retained in the source. Because
the input datasets and trained weights are not distributed here, complete
numerical reproduction requires obtaining the cited datasets and following the
same preprocessing described in the manuscript and Supporting Information.

## Citation

Please cite the associated article when using this code. Publication details
and a DOI will be added when available.

## Contact

Questions about the code may be directed to the corresponding author,
Dileepkumar R.
