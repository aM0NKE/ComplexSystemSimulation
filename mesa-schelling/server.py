import mesa

from model import Schelling


def get_model_stats(model):
    """
    Display a text of some general model statistics.
    """
    
    return (
            f"Happy agents: {model.happy}",
            f"\nHappy distribution: {model.happy_dist}",
            f"\nTotal wealth: {model.total_wealth}",
            f"\nWealth distribution: {model.wealth_dist}",
            f"\nTotal avg. cluster size: {model.total_avg_cluster_size}",
            f"\nCluster size data: {model.cluster_data}",
            f"\nPercolation data: {model.percolation_data}",
        )
        
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
    "height": mesa.visualization.Slider("Grid height", 100, 10, 100, 10),
    "width": mesa.visualization.Slider("Grid width", 100, 10, 100, 10),
    "density": mesa.visualization.Slider("Agent density", 0.97, 0.1, 1.0, 0.01),
    "fixed_areas_pc": mesa.visualization.Slider("Fixed area density", 0.0, 0.0, 0.2, 0.025),
    #"minority_pc": mesa.visualization.Slider("Fraction minority", 0.2, 0.00, 1.0, 0.05),
    "homophily": mesa.visualization.Slider("Homophily", 3, 0, 8, 1),
    "cluster_threshold": mesa.visualization.Slider("cluster_threshold", 4, 1, 8, 1),
}

# Define graphic elements
canvas_element = mesa.visualization.CanvasGrid(schelling_draw, 100, 100, 1000, 1000)
happy_chart = mesa.visualization.ChartModule([{"Label": "happy", "Color": "Black"}])
total_wealth_chart = mesa.visualization.ChartModule([{"Label": "total_wealth", "Color": "Black"}])
total_cluster_size_chart = mesa.visualization.ChartModule([{"Label": "total_avg_cluster_size", "Color": "Black"}])

# Create the server, and pass the grid and the graph
server = mesa.visualization.ModularServer(
    Schelling,
    [canvas_element, get_model_stats, total_cluster_size_chart, happy_chart, total_wealth_chart],
    "Schelling",
    model_params,
)