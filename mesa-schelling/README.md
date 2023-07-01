# Schelling Segregation Model (+added Economic Component)

## Installation

To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install -r requirements.txt
```

## How to Run

To run the model interactively, run ``mesa runserver`` in this directory. e.g.

```
    $ mesa runserver
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press Reset, then Run.

To view and run some model analyses, launch the IPython Notebook and open ``analysis/final_analysis.ipynb``. Here you will find all of the answers to our research questions.

## How to Run without the GUI

To run the model with the grid displayed as an ASCII text, run `python run_ascii.py` in this directory. This script was mainly used for debugging. 

## Files

* ``model.py``: Contains the agent class, and the overall model class.
* ``helper_functions/``: Contains functions to calculate model stats.
* ``run.py``: Launches a model visualization server.
* ``server.py``: Defines classes for visualizing the model in the browser via Mesa's modular server, and instantiates a visualization server.
* ``run_ascii.py``: Run the model in text mode (for debugging).
* ``analysis/final_analysis.ipynb``: Notebook containing the answers to our research questions.
* ``analysis/plots/``: Folder containing all of the plots that answer our research questions.
* ``analysis/archive/``: Folder containing all of the testing code we have created during our project, most of it has been incorporated in our final implementation. 
