import mesa

class SchellingAgent(mesa.Agent):
    """
    Agent class of the Schelling segregation agent.
    """

    def __init__(self, pos, model, agent_type):
        """
        Create a new Schelling agent.

        Args:
           unique_id: Unique identifier for the agent.
           pos (x, y): Agent initial location.
           model: Model reference. 
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type

    def move(self):
        """
        If the agent is unhappy, move to a new location.
        """

        # Determine the agent's similar neighbors
        similar = 0
        for neighbor in self.model.grid.iter_neighbors(self.pos, True):
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1

    def step(self):
        self.move()


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, width=20, height=20, density=0.8, minority_pc=0.2, homophily=3):
        """ """

        # Set parameters
        self.width = width
        self.height = height
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily

        # Set up model objects
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        # Set up model statistics
        self.happy = 0
        self.datacollector = mesa.DataCollector(
            {"happy": "happy"},  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
        )
        self.datacollector.collect(self)

        # Variable to halt model
        self.running = True

        # Set up agents
        self.setup_agents()

    def setup_agents(self):
        """
        Places agents on the grid. 

        Note: We use a grid iterator that returns
              the coordinates of a cell as well as
              its contents. (coord_iter)
        """
        
        # Loop over each cell in the grid
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            
            # Place an agent nased on the density
            if self.random.random() < self.density:
                # Determine if the agent is a minority
                if self.random.random() < self.minority_pc:
                    agent_type = 1
                else:
                    agent_type = 0

                # Create a new agent
                agent = SchellingAgent((x, y), self, agent_type)
                # Add the agent to the grid
                self.grid.place_agent(agent, (x, y))
                # Add the agent to the scheduler
                self.schedule.add(agent)

    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """

        # Reset counter of happy agents
        self.happy = 0

        # Advance each agent by one step
        self.schedule.step()

        # Collect data
        self.datacollector.collect(self)

        # Halt if no unhappy agents
        if self.happy == self.schedule.get_agent_count():
            self.running = False