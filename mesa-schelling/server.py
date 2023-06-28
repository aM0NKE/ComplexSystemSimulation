import mesa

from model import Schelling


def happy_stats(model):
    """
    Display a text of some general model statistics.
    """
    return f"Happy agents: {model.happy}"
            # , f"\nHappy distribution: {model.happy_dist}"

def wealth_stats(model):
    return f"\nTotal wealth: {model.total_wealth}"
            # , f"\nWealth distribution: {model.wealth_dist}"

def cluster_stats(model):
    return f"\nTotal avg. cluster size: {model.total_avg_cluster_size}"
            # , f"\nCluster size data: {model.cluster_data}"

def percolation_stats(model):
    return f"\nPercolation data: {model.percolation_data}", f"\nPercolation boolean: {model.boolean_percolation}"
     
def schelling_draw(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    
    portrayal = {"Shape": "circle", "r": .8, "Filled": "true", "Layer": 0}
    
    if agent.type == -1:
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Color"] = ["#5A5A5A"]
    else:
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFA500", "#FF00FF", "#00FFFF", "#FFFF00", "#800080", "#008000", "#FFC0CB"]
        num_colors = len(colors)
        agent_type = agent.type % num_colors
        portrayal["Color"] = colors[agent_type]

    return portrayal

# Define model parameters
model_params = {
    "size": mesa.visualization.Slider("Grid size", 100, 10, 100, 10),
    "density": mesa.visualization.Slider("Agent density", 0.9, 0.1, 1.0, 0.01),
    "homophily": mesa.visualization.Slider("Homophily", 4, 0, 8, 1),
    "alpha": mesa.visualization.Slider("Alpha", 0.1, 0, .99, 0.01),
    "cluster_threshold": mesa.visualization.Slider("Cluster size threshold", 4, 1, 100, 1),
    "stopping_threshold": mesa.visualization.Slider("Stopping threshold", 5, 1, 20, 1),
    "fixed_areas_pc": mesa.visualization.Slider("Fixed area density (Old Idea)", 0.0, 0.0, 0.2, 0.025),
    "server": True,
}

# Define graphic elements
canvas_element = mesa.visualization.CanvasGrid(schelling_draw, 100, 100, 1000, 1000)
happy_chart = mesa.visualization.ChartModule([  {"Label": "happy", "Color": "Black"},
                                                {"Label": "happy_t0", "Color": "#FF0000"},
                                                {"Label": "happy_t1", "Color": "#00FF00"},
                                                {"Label": "happy_t2", "Color": "#0000FF"},
                                                {"Label": "happy_t3", "Color": "#FFA500"},
                                                {"Label": "happy_t4", "Color": "#FF00FF"},
                                                {"Label": "happy_t5", "Color": "#00FFFF"},
                                                {"Label": "happy_t6", "Color": "#FFFF00"},
                                                {"Label": "happy_t7", "Color": "#800080"},
                                                {"Label": "happy_t8", "Color": "#008000"},
                                                {"Label": "happy_t9", "Color": "#FFC0CB"},
                                            ])
wealth_chart = mesa.visualization.ChartModule([ {"Label": "total_wealth", "Color": "Black"},
                                                {"Label": "wealth_t0", "Color": "#FF0000"},
                                                {"Label": "wealth_t1", "Color": "#00FF00"},
                                                {"Label": "wealth_t2", "Color": "#0000FF"},
                                                {"Label": "wealth_t3", "Color": "#FFA500"},
                                                {"Label": "wealth_t4", "Color": "#FF00FF"},
                                                {"Label": "wealth_t5", "Color": "#00FFFF"},
                                                {"Label": "wealth_t6", "Color": "#FFFF00"},
                                                {"Label": "wealth_t7", "Color": "#800080"},
                                                {"Label": "wealth_t8", "Color": "#008000"},
                                                {"Label": "wealth_t9", "Color": "#FFC0CB"},
                                            ])
cluster_chart = mesa.visualization.ChartModule([{"Label": "total_avg_cluster_size", "Color": "Black"},
                                                {"Label": "cluster_t0", "Color": "#FF0000"},
                                                {"Label": "cluster_t1", "Color": "#00FF00"},
                                                {"Label": "cluster_t2", "Color": "#0000FF"},
                                                {"Label": "cluster_t3", "Color": "#FFA500"},
                                                {"Label": "cluster_t4", "Color": "#FF00FF"},
                                                {"Label": "cluster_t5", "Color": "#00FFFF"},
                                                {"Label": "cluster_t6", "Color": "#FFFF00"},
                                                {"Label": "cluster_t7", "Color": "#800080"},
                                                {"Label": "cluster_t8", "Color": "#008000"},
                                                {"Label": "cluster_t9", "Color": "#FFC0CB"},
                                            ])

# Create the server, and pass the grid and the graph
server = mesa.visualization.ModularServer(
    Schelling,
    [canvas_element, percolation_stats, wealth_stats, wealth_chart, cluster_stats, cluster_chart, happy_stats, happy_chart],
    "Schelling",
    model_params,
)