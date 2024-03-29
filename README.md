# Experimental Data Modeling and Analysis
Project for modeling and analysis of experimental data from Experimental Data Modeling and Analysis Course.

The project is a Python application that allows to model, process, and analyze both 1D and 2D signals. For example, you can simulate noise / cardiogram and other effects, highlight edges and details in images, change
the stress in speech, etc.

# Project Structure
 .  
├── src/  
        ├── \_\_init\_\_.py  
        ├── analysis.py                        # Contains statistical functions  
        ├── data.py                          # Input data  
        ├── model.py                       # Model trends, noise, shifts, outliers, harmonic processes, etc.  
        ├── processing.py                # Process inputs by removing trends, shifts, spikes, noise, apply gradational transformations, etc.  
├── main.py                                 # Space to work with data and run code  
├── examples/                             # Folder contains images showing inputs and outputs  
└── README.md
