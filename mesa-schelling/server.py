import mesa

from model import Schelling


def get_happy_agents(model):
    """
    Display a text count of how many happy agents there are.
    """
    
    return f"Happy agents: {model.happy}"

def get_cluster_info(model):
    """
    Display a text count of how many happy agents there are.
    """
    
    return f"Cluster summary: {model.cluster_data}"

def schelling_draw(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.6, "Filled": "true", "Layer": 0}
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#FF00FF", "#00FFFF", "#FFFF00", "#800080", "#008000", "#FFC0CB"]

    num_colors = len(colors)
    agent_type = agent.type % num_colors
    portrayal["Color"] = colors[agent_type]

    return portrayal


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


model_params = {
    "height": mesa.visualization.Slider("Grid height", 100, 10, 100, 10),
    "width": mesa.visualization.Slider("Grid width", 100, 10, 100, 10),
    "density": mesa.visualization.Slider("Agent density", 0.95, 0.1, 1.0, 0.01),
    "fixed_areas_pc": mesa.visualization.Slider("Fixed area density", 0.0, 0.0, 0.2, 0.025),
    #"minority_pc": mesa.visualization.Slider("Fraction minority", 0.2, 0.00, 1.0, 0.05),
    "homophily": mesa.visualization.Slider("Homophily", 3, 0, 8, 1),
}

canvas_element = mesa.visualization.CanvasGrid(schelling_draw, 100, 100, 1000, 1000)
happy_chart = mesa.visualization.ChartModule([{"Label": "happy", "Color": "Black"}])


server = mesa.visualization.ModularServer(
    Schelling,
    [canvas_element, get_happy_agents, happy_chart, get_cluster_info],
    "Schelling",
    model_params,
)