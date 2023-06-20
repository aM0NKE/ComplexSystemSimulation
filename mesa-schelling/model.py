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

    def __init__(self, width=100, height=100, density=0.8, N=2, pop_weights=[0.5, 0.5], homophily=3):
        """ """

        # Set parameters
        self.width = width
        self.height = height
        self.density = density
        self.N = N
        self.pop_weights = pop_weights
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
        
        # Check if population fractions are provided
        if self.pop_weights is None:
            raise ValueError("Population fractions must be specified.")

        # Check if the number of population fractions matches the number of populations
        if len(self.pop_weights) != self.N:
            raise ValueError("Number of population fractions must match the number of populations.")
        
        # Check if population fractions add up to 1
        if round(sum(self.pop_weights), 4) != 1.0:
            raise ValueError("The population fractions do not add up to 1")
        
        # Loop over the population fractions and validate the values
        for weight in self.pop_weights:
            if weight < 0 or weight > 1:
                raise ValueError("Population fractions must be between 0 and 1.")

        # Loop over each cell in the grid
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            
            # Place an agent nased on the density
            if self.random.random() < self.density:
                # Determine if the agent is a minority
                agent_type = self.random.choices(
                    population=range(self.N),
                    weights=self.pop_weights,)[0]

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