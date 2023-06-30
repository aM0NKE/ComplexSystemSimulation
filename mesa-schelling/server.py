import mesa

from model import Schelling


# Define a bunch of functions that return text to be displayed in the GUI
def happy_stats(model):
    return f"Total happy agents: {model.total_happy}"

def wealth_stats(model):
    return f"\nTotal wealth: {model.total_wealth}"

def cluster_stats(model):
    return f"\nTotal avg. cluster size: {model.total_avg_cluster_size}"

def percolation_per_pop(model):
    return f"\nPercolation per population [vertical, horizontal]: {model.percolation_per_pop}"
     
def percolation_system(model):
    return f"\nPercolation system: {model.percolation_system}"
     
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
    "density": mesa.visualization.Slider("Agent density", 0.9, 0.05, .99, 0.05),
    "homophily": mesa.visualization.Slider("Homophily", 4, 0, 8, 1),
    "alpha": mesa.visualization.Slider("Alpha", 0.95, 0.5, 10, 0.1),
    "cluster_threshold": mesa.visualization.Slider("Cluster size threshold", 5, 4, 100, 5),
    "stopping_threshold": mesa.visualization.Slider("Stopping threshold", 10, 0, 50, 5),
    "fixed_areas_pc": mesa.visualization.Slider("Fixed area density (Old Idea)", 0.0, 0.0, 0.2, 0.025),
    "server": True,
}

# Define graphic elements
canvas_element = mesa.visualization.CanvasGrid(schelling_draw, 100, 100, 1000, 1000)
happy_chart = mesa.visualization.ChartModule([  {"Label": "happy_t0", "Color": "#FF0000"},
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
wealth_chart = mesa.visualization.ChartModule([ {"Label": "wealth_t0", "Color": "#FF0000"},
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
    [canvas_element, percolation_per_pop, percolation_system, wealth_stats, wealth_chart, cluster_stats, cluster_chart, happy_stats, happy_chart],
    "Schelling",
    model_params,
)