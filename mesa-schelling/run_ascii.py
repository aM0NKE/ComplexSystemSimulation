import mesa

from model import Schelling


class SchellingTextVisualization(mesa.visualization.TextVisualization):
    """
    ASCII visualization for schelling model
    """

    def __init__(self, model):
        """
        Create new Schelling ASCII visualization.
        """
        self.model = model

        grid_viz = mesa.visualization.TextGrid(self.model.grid, self.print_ascii_agent)
        self.elements = [grid_viz]

    @staticmethod
    def print_ascii_agent(a):
        """
        Minority agents are X, Majority are O.
        """
        if a.type == 0:
            return "O"
        if a.type == 1:
            return "X"
        if a.type == -1: # Fixed objects
            return "■"

if __name__ == "__main__":
    model_params = {
        "size": 20,
        "density": 0.8,
        "fixed_areas_pc": 0.0,
        "pop_weights":[0.8, 0.2],
        "homophily": 5,
        "cluster_threshold": 0,
        "alpha": .95,
        "stopping_threshold": 5,
        'stats': True,
        "server": True
    }

    model = Schelling(**model_params)
    viz = SchellingTextVisualization(model)
    for i in range(50):
        print("Step:", i)
        viz.step()
        print("---")