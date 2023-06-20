import mesa

from model import Schelling


def get_happy_agents(model):
    """
    Display a text count of how many happy agents there are.
    """
    return f"Happy agents: {model.happy}"


def schelling_draw(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.6, "Filled": "true", "Layer": 0}

    if agent.type == 0:
        portrayal["Color"] = "#FF0C00" # Red
        #portrayal["stroke_color"] = "#000000"
    elif agent.type == 1:
        portrayal["Color"] = "#1B00FF" # Blue
        #portrayal["stroke_color"] = "#000000"
    elif agent.type == 2:
        portrayal["Color"] = "#00CB28" # Green
        #portrayal["stroke_color"] = "#000000"
    elif agent.type == 3:
        portrayal["Color"] = "#FF80DE" # Pink
        #portrayal["stroke_color"] = "#000000"
    return portrayal

model_params = {
    "height": mesa.visualization.Slider("Grid height", 20, 10, 100, 10),
    "width": mesa.visualization.Slider("Grid width", 20, 10, 100, 10),
    "density": mesa.visualization.Slider("Agent density", 0.8, 0.1, 1.0, 0.1),
    #"minority_pc": mesa.visualization.Slider("Fraction minority", 0.2, 0.00, 1.0, 0.05),
    "homophily": mesa.visualization.Slider("Homophily", 3, 0, 8, 1),
}

canvas_element = mesa.visualization.CanvasGrid(schelling_draw, 100, 100, 1000, 1000)
happy_chart = mesa.visualization.ChartModule([{"Label": "happy", "Color": "Black"}])


server = mesa.visualization.ModularServer(
    Schelling,
    [canvas_element, get_happy_agents, happy_chart],
    "Schelling",
    model_params,
)