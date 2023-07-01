# Schelling Segregation Model (+ added economic component)

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

To view and run some model analyses, launch the IPython Notebook and open ``analysis/final_analysis.ipynb``. Visualizing the analysis also requires [matplotlib](http://matplotlib.org/).

## How to Run without the GUI

To run the model with the grid displayed as an ASCII text, run `python run_ascii.py` in this directory.

## Files

* ``run.py``: Launches a model visualization server.
* ``run_ascii.py``: Run the model in text mode.
* ``model.py``: Contains the agent class, and the overall model class.
* ``server.py``: Defines classes for visualizing the model in the browser via Mesa's modular server, and instantiates a visualization server.
* ``analysis/final_analysis.ipynb``: Notebook containing the results to our sub-research questions.
* ``helper_functions/``: Contains functions to calculate model stats.