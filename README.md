# IMU-Fundamentals
Code repository outlining basic concepts related to using inertial measurement units (IMUs) for human motion analysis.

## Running in Python

### Environment setup 
Python users must setup an appropriate environment to run the IMU-Fundamentals notebook. This can be run in any IDE that supports Jupyter notebooks (such as Jupyter Notebook, JupyterLab, VS Code, etc.) after cloning the repository and changing your directory accordingly. 

1. Create environment: `conda env create -f environment.yml`
2. Activate environment: `conda activate IMU_Funds`
3. Open `IMU_Fundamentals.ipynb` and run all cells in order

### Running through the tutorial
Users can find the tutorial within `IMU_Fundamentals.ipynb`. All cells should be run in order. The `table2zoo_data/` folder will be created automatically when you run the cells. All functions that are used throughout the tutorial can be inspected within their respective .py files within the `utils/` folder.

## Project Structure
```
IMU-Fundamentals/
├── IMU_Fundamentals.ipynb    # Main tutorial notebook
├── environment.yml            # Conda environment file
├── README.md                  # This file
└── utils/                     # Utility functions
```

## Requirements
- Python 3.11+
- Conda (for environment management)

## Author & License
Joshua Lowery - MSc Biomechanics Student @ McGill Univeristy's MOTION Laboratory. 

GPL-3.0 license 
   
