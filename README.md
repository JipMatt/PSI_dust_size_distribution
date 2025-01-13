# Code underlying the manuscript "Polydisperse Formation of Planetesimals"

This repository contains the setup scripts required for the project titled [Polydisperse Formation of Planetesimals The dust size distribution in clumps](URL), which is documented in the accompanying research paper. 

This reposetory is openly avialble on `4TU.ResearchData` under the name: [Code underlying the manuscript "Polydisperse Formation of Planetesimals"](http://doi.org/10.4121/528086f3-b288-482f-ab45-3c5ccb18a293)

The following sections describe the structure of the repository, how to install dependencies, and how to set up the environment to use the scripts and data.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Description](#data-description)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Paper](#paper)
- [License](#license)
- [Contact](#contact)

## Project Structure

```plaintext
|-- data_prep.py        # Script for data preprocessing raw data from FARGO3D
|-- plots_paper.py      # Script generates the plots of manuscript from the raw data and processed data.
|    
|-- vistools.py         # Class to read in raw data in .dat format
|-- polydust.py         # Class holding a size (or stopping time) distribution
|
|-- README.md           # This file
|-- requirements.txt    # List of Python dependencies
|-- environment.yml     # Conda environment file (if using Conda)
```

## Prerequisites

Before running the scripts, ensure that you have the following software installed:

- **Python 3.x** (or the version specified in the `requirements.txt` or `environment.yml`)
- **FARGO3d** [FARGO3D](`https://github.com/FARGO3D/fargo3d`)
- **`pip` or `conda`**
- **Git** (for cloning a repository [`psitools`](https://github.com/psitools/psitools-public))


## Installation

### Step 1: Clone the Repository

If you haven't cloned the repository, you can do so using the following command:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Install Dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, using `conda`:

```bash
conda env create -f environment.yml
conda activate your-env-name
```

Another repository psitools has to be cloned through git using:

```bash
git clone https://github.com/psitools/psitools-public
pip install -e  ~/path/to/psitools-public
```

## Data Description

The data that support the findings of this study are openly available in `4TU.ResearchData` under the name: [Data underlying the manuscript "Polydisperse Formation of Planetesimals"](http://doi.org/10.4121/8f0a99b2-7d55-4e05-aa8f-345cab38ff38).

### Directory Structure

The data is stored in the `/data` directory. This is further divided into two subdirectories:

1. **Raw Data (`/data/raw`)**: This folder contains the raw output of the specific snapshots from FARGO3D used in the corresponding paper. Each dataset has been named according to the convention of FARGO3D`[dataset_name].dat`. Although, due to space constraints only specific snapshots relevant to the paper are stored. The raw data can be reproduced by running the `psi/createsetup.py` to make a setup and run this setup with FARGO3D.

2. **Processed Data (`/data/processed`)**: This folder contains the preprocessed data ready for analysis. The preprocessing steps are defined in the script `data_prep.py`. Each dataset has been named according to the convention of FARGO3D`[res[resolution]_poly[mono/ndust]_ts([stokesrange])_k[wavenumber/WN]_[additinal_parameter]_[amp/ts_disc/ts_cont/hist].csv`, with `_amp.csv` being the density/amplitude series (rho,t), `_ts_disc.csv` the sizedistribution at specific stokes numbers at different snapshots, `_ts_cont.csv` the continous size distribution, and `_hist.csv` the PDF of the density distribution. Additianly, there is a file `growthrates_ts.csv` containg the growth rates at different stokes numbers calucated with `psitools`.

### File Descriptions

- `data/raw/dataset1.dat`: All the outputs of FARGO3D for the different fluids at a specific snapshot.
- `data/processed/processed.csv`: Processed version of `dataset1.csv` used for the final analysis.

### Data Preprocessing

The preprocessing steps, including data cleaning, feature selection, and transformation, are executed by the `data_prep.py` script. Modify this script to adapt to your specific data if necessary.

You can get timeseries and sizedistribution from  FARGO3D simulation has run run the preprocessing script to transform the raw data into a form suitable for analysis:

```bash
python scripts/data_prep.py --input "All snapshot of a FARGO3D simulation run" --output run_name_amp.csv run_name_ts_disc.csv run_name_ts_cont.csv  run_name_hist.csv 
```

## Usage

Running the analysis script you reproduce the plot in the paper.

Example:

```bash
# Example command to run an analysis
python scripts/plots_paper.py --input data/processed/dataset1_processed.csv --output name_figure.png
```

### Important Notes

- Ensure the paths to datasets are correctly defined in the scripts.
- Modify the parameters inside the scripts (e.g., hyperparameters, input file paths) if necessary.

## Paper

    Polydisperse Formation of Planetesimals: The dust size distribution in clumps
    Matthijsse, Jip; Aly, Hossam; Paardekooper, Sijme-Jan
    Astronomy & Astrophysics Journal,
    DOI: 
    ADS: 
    arXiv:

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

## Contact

* Jip Matthijsse
* Hossam Aly
* Sijme-Jan Paardekooper

For any inquiries or questions, please contact:

- **Author Name**: Jip Matthijsse
- **Email**:       j.p.matthijsse@tudelft.nl
- **Institution**: Technical University Delft
 