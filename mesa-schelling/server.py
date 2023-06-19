from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement

from model import SchellingModel


class HappyElement(TextElement):
    '''
    Display a text count of how many happy agents there are.
    '''

    def render(self, model):
        return "Happy agents: " + str(model.happy)


def schelling_draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}

    if agent.type == 0:
        portrayal["Color"] = "Red"
    else:
        portrayal["Color"] = "Blue"
    return portrayal

happy_element = HappyElement()
canvas_element = CanvasGrid(schelling_draw, 100, 100, 500, 500)
happy_chart = ChartModule([{"Label": "happy", "Color": "Black"}])

server = ModularServer(SchellingModel,
                       [canvas_element, happy_element, happy_chart],
                       "Schelling’s Segregation Model",
                       100, 100, 0.8, 0.2, 4)
