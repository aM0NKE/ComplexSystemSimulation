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
        happy_viz = mesa.visualization.TextData(self.model, "happy")
        total_wealth_viz = mesa.visualization.TextData(self.model, "total_wealth")
        self.elements = [grid_viz, happy_viz, total_wealth_viz]

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
        # Agent density, from 0.8 to 1.0
        "density": 0.8,
        # Fraction minority, from 0.2 to 1.0
        # "minority_pc": 0.2,
        # Fixed area density from 0.0 to 0.2
        "fixed_areas_pc": 0.0,
        # Number of population groups, from 2 to 4
        "pop_weights":[0.8, 0.2],
        # Homophily, from 3 to 8
        "homophily": 3,
        # Cluster threshold, from 4 to 8
        "cluster_threshold": 20,
        "alpha": .95,
        "stopping_threshold": 5,
        "server": False
    }

    model = Schelling(**model_params)
    viz = SchellingTextVisualization(model)
    for i in range(10):
        print("Step:", i)
        viz.step()
        print("---")