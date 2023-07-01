# Import modules
import mesa
import random

# Import helper functions
from helper_functions.segregation_clusters import *
from helper_functions.wealth_clusters import *
from helper_functions.percolation import *

class SchellingAgent(mesa.Agent):
    """
    Agent class of the Schelling segregation agent.
    """

    def __init__(self, pos, model, agent_type, init_wealth):
        """
        Create a new Schelling agent.

        Args:
            unique_id: Unique identifier for the agent.
            pos (x, y): Agent initial location.
            model: Model reference. 
            agent_type: Indicator for the agent's type (maximum of 10).
            init_wealth: Initial wealth of the agent.
        """

        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.wealth = init_wealth

    def calculate_avg_neighbor_wealth(self, neighbors):
        """
        Calculates the average wealth of the agent's neighbors.

        Args:
            neighbors: List of the agent's neighbors.
        """

        # Initialize counters
        cnt_neighbors = 0
        wealth_neighbors = 0

        # Iterate over all neighbors in moore neighborhood
        for neighbor in neighbors:

            # Skip fixed objects
            if neighbor.type == -1:
                continue 

            # Update counters
            cnt_neighbors += 1
            wealth_neighbors += neighbor.wealth

        # Calculate average wealth of neighbors
        avg_neighbor_wealth = wealth_neighbors / cnt_neighbors
        return avg_neighbor_wealth

    def update_wealth(self):
        """
        Updates the agent's wealth based on the average wealth of its neighbors.
        """

        # Find the agent's neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, True)

        # If an agent has neighbors, update its wealth
        if len(neighbors) > 0 and self.is_happy():

            # Find avg. wealth of agent's neighbors
            avg_neighbor_wealth = self.calculate_avg_neighbor_wealth(neighbors)
            
            # Economic rules V2
            # Check if agent's wealth is within the range of its neighbors' wealth
            if (1 - self.model.alpha) <= (self.wealth / avg_neighbor_wealth) <= 1:
                self.wealth = 0.5 * self.wealth + 0.5 * avg_neighbor_wealth
            else:
                self.wealth = self.wealth

            # Economic rules V1
            # Check if agent's wealth is less than the avg. wealth of its neighbors
            # if self.wealth < avg_neighbor_wealth:
            #     self.wealth = 0.5 * self.wealth + 0.5 * avg_neighbor_wealth

        # If an agent has no neighbors, keep its wealth
        else:
            self.wealth = self.wealth
        
        # Update the model's total wealth
        self.model.total_wealth += self.wealth

        # Update the wealth distribution
        self.model.wealth_dist[self.type] += self.wealth

        # Update individual wealth per agent type attributes for visualiaztion
        var_key = f'wealth_t{self.type}'
        setattr(self.model, var_key, getattr(self.model, var_key)+self.wealth)

    def is_happy(self):
        """
        Determines whether the agent is happy or not.
        """

        # Find the agent's neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, True)

        # Find the agent's similar neighbors
        similar = 0
        for neighbor in neighbors:
            if neighbor.type == self.type:
                similar += 1

        # If the agent is unhappy, return False
        if similar < self.model.homophily:
            return False

        # If the agent is happy, return True
        else:
            return True

    def move(self):
        """
        If the agent is unhappy, move to a new location.
        """

        # If unhappy, move:
        if self.is_happy() == False:
            self.model.grid.move_to_empty(self)
        else:
            # Update the model's happy count
            self.model.total_happy += 1
            
            # Update the model's happy distribution
            self.model.happy_dist[self.type] += 1

            # Update individual happy per agent type attributes for visualiaztion
            var_key = f'happy_t{self.type}'
            setattr(self.model, var_key, getattr(self.model, var_key)+1)

    def step(self):
        if self.type != -1: # Don't move fixed objects
            self.move()
            self.update_wealth()


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, size=100, density=0.9, fixed_areas_pc=0.0, 
                 pop_weights=[0.5, 0.3, 0.2], homophily=4, cluster_threshold=4, 
                 alpha=0.95, stopping_threshold=5, stats=False, server=False):
        """ 
        Initialize the Schelling model.

        Args:
            size: Size of the grid (size x size).
            density: The proportion of the grid to be occupied by agents.
            fixed_areas_pc: The proportion of the grid to be occupied by fixed areas.
            pop_weights: The proportion of each agent type in the population.
            homophily: The minimum number of similar neighbors an agent needs to be happy.
            cluster_threshold: The minimum number of agents together to count as a cluster.
            alpha: The parameter to check if agent's wealth is within the range of its neighbors' wealth.
            stopping_threshold: If the model's happy count does not change by this threshold, stop the model.
            stats: Boolean indicating whether to calculate the expensive model statistics or not.
            server: Boolean indicating whether the model is run on a server or not.
                    Note: This was added to avoid stopping criterion errors when running the model on a server.
        """

        # Set parameters
        self.size = size
        self.density = density
        self.fixed_areas_pc = fixed_areas_pc
        self.pop_weights = pop_weights
        self.N = len(pop_weights)
        self.homophily = homophily
        self.cluster_threshold = cluster_threshold
        self.alpha = alpha
        self.stopping_threshold = stopping_threshold

        # Set up model objects
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(size, size, torus=True)

        # Set up model statistics (excuse us for the long list of attributes)
        # Note: All these variables were added so that they could be included
        #       in the server visualization of the model.
        self.total_happy = 0
        self.happy_dist = {i: 0 for i in range(self.N)}
        self.happy_t0 = 0
        self.happy_t1 = 0
        self.happy_t2 = 0
        self.happy_t3 = 0
        self.happy_t4 = 0
        self.happy_t5 = 0
        self.happy_t6 = 0
        self.happy_t7 = 0
        self.happy_t8 = 0
        self.happy_t9 = 0

        self.total_wealth = 0
        self.wealth_dist = {i: 0 for i in range(self.N)}
        self.wealth_t0 = 0
        self.wealth_t1 = 0
        self.wealth_t2 = 0
        self.wealth_t3 = 0
        self.wealth_t4 = 0
        self.wealth_t5 = 0
        self.wealth_t6 = 0
        self.wealth_t7 = 0
        self.wealth_t8 = 0
        self.wealth_t9 = 0

        self.total_avg_cluster_size = 0.0
        self.cluster_sizes_per_pop = {}
        self.cluster_summary = {}
        self.cluster_t0 = 0
        self.cluster_t1 = 0
        self.cluster_t2 = 0
        self.cluster_t3 = 0
        self.cluster_t4 = 0
        self.cluster_t5 = 0
        self.cluster_t6 = 0
        self.cluster_t7 = 0
        self.cluster_t8 = 0
        self.cluster_t9 = 0

        # Order parameters
        self.percolation_per_pop = {}
        self.percolation_system = 0
        self.stats = stats # Boolean indicating whether to calculate the expensive model statistics or not.
        self.segregation_coefficient = 0.0
        self.half_time = 0

        # Define datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "total_happy": lambda m: m.total_happy,
                "happy_dist": lambda m: m.happy_dist.copy(),
                "happy_t0": lambda m: m.happy_t0, 
                "happy_t1": lambda m: m.happy_t1, 
                "happy_t2": lambda m: m.happy_t2, 
                "happy_t3": lambda m: m.happy_t3, 
                "happy_t4": lambda m: m.happy_t4,
                "happy_t5": lambda m: m.happy_t5,
                "happy_t6": lambda m: m.happy_t6,
                "happy_t7": lambda m: m.happy_t7,
                "happy_t8": lambda m: m.happy_t8,
                "happy_t9": lambda m: m.happy_t0,  
                "total_wealth": lambda m: m.total_wealth,
                "wealth_dist": lambda m: m.wealth_dist.copy(),
                "wealth_t0": lambda m: m.wealth_t0,
                "wealth_t1": lambda m: m.wealth_t1,
                "wealth_t2": lambda m: m.wealth_t2,
                "wealth_t3": lambda m: m.wealth_t3,
                "wealth_t4": lambda m: m.wealth_t4,
                "wealth_t5": lambda m: m.wealth_t5,
                "wealth_t6": lambda m: m.wealth_t6,
                "wealth_t7": lambda m: m.wealth_t7,
                "wealth_t8": lambda m: m.wealth_t8,
                "wealth_t9": lambda m: m.wealth_t9,
                "total_avg_cluster_size": lambda m: m.total_avg_cluster_size,
                "cluster_sizes_per_pop": lambda m: m.cluster_sizes_per_pop.copy(),
                "cluster_summary": lambda m: m.cluster_summary.copy(),
                "cluster_t0": lambda m: m.cluster_t0,
                "cluster_t1": lambda m: m.cluster_t1,
                "cluster_t2": lambda m: m.cluster_t2,
                "cluster_t3": lambda m: m.cluster_t3,
                "cluster_t4": lambda m: m.cluster_t4,
                "cluster_t5": lambda m: m.cluster_t5,
                "cluster_t6": lambda m: m.cluster_t6,
                "cluster_t7": lambda m: m.cluster_t7,
                "cluster_t8": lambda m: m.cluster_t8,
                "cluster_t9": lambda m: m.cluster_t9,
                "percolation_per_pop": lambda m: m.percolation_per_pop.copy(), 
                "percolation_system": lambda m: m.percolation_system, 
                "segregation_coefficient": lambda m: m.segregation_coefficient,
                "half_time": lambda m: m.half_time, 
            },
            agent_reporters={
                # For testing purposes, agent's individual x and y
                "x": lambda a: a.pos[0], 
                "y": lambda a: a.pos[1], 
            }
        )
        self.datacollector.collect(self)

        # Variable to halt model
        self.running = True
        self.stopping_cnt = 0
        self.happy_max = 0

        # Parameter added to prevent stopping error when running model on server.
        self.server = server

        # Add fixed cells
        self.fixed_cells = []
        if fixed_areas_pc > 0:
            self.add_fixed_cells(fixed_areas_pc)

        # Set up agents
        self.setup_agents()

        # Check if the model has been initialized correctly
        self.check_initialization()

    def check_initialization(self):
        """
        Checks if the model has been initialized correctly.
        """
        
        assert 0 < self.size <= 200, "Grid size must be between 0 and 200."
        assert 0 < self.grid.width <= 200, "Grid was not initialized correctly."
        assert 0 < self.grid.height <= 200, "Grid was not initialized correctly."
        assert 0 < self.density < 1, "Density must be between 0 and 1."
        assert 0 <= self.fixed_areas_pc <= 0.5, "Proportion of fixed areas must be between 0 and 0.5."
        assert 0 < self.N <= 10, "Number of agents must be between 0 and 10."
        assert round(sum(self.pop_weights), 4) <= 1.0, "Population fractions must add up to 1."
        assert 0 <= self.homophily <= 8, "Tolerance threshold must be between 0 and 8."
        assert 0 <= self.cluster_threshold <= 1000, "Cluster threshold must be between 0 and 1000."
        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1."
        assert self.grid is not None, "Grid was not initialized correctly, please try again."
        assert self.schedule is not None, "Schedule was not initialized correctly, please try again."
        assert self.datacollector is not None, "Datacollector was not initialized correctly, please try again."
        assert self.running is True, "Model was not initialized correctly, please try again."
        assert len(self.schedule.agents) != self.size * self.size, "No empty cells in the grid, please try again."
        
    def add_fixed_cells(self, fixed_areas_pc):
        """
        Adds fixed cells to the grid by randomly creating clusters 
        of fixed cells and placing them on the grid.

        Args:
            fixed_areas_pc: Proportion of fixed cells in the grid.

        Note: Fixed cells are represented as an agent of type -1.
        """

        # Calculate the number of fixed cells to add to the grif
        num_fixed_cells = int(fixed_areas_pc * self.size * self.size)

        # Generate clusters of fixed cells
        cluster_centers = self.generate_cluster_centers(num_fixed_cells)
        for center in cluster_centers:
            
            # Generate a cluster of fixed cells around the center
            cluster = self.generate_cluster(center)
            for cell in cluster:
            
                # Skip if the cell is already occupied
                if cell in [agent.pos for agent in self.fixed_cells]:
                    continue

                # Create a new fixed cell agent
                agent = SchellingAgent(cell, self, -1, 0)

                # Add the agent to the grid
                self.grid.place_agent(agent, cell)
                self.fixed_cells.append(agent)

    def generate_cluster_centers(self, num_centers):
        """
        Helper function that generates random cluster centers.

        Args:
            num_centers: Number of cluster centers to generate.

        Returns:
            List of cluster center coordinates.
        """

        cluster_centers = []
        for _ in range(num_centers):
            x = self.random.randrange(self.size)
            y = self.random.randrange(self.size)
            cluster_centers.append((x, y))
        return cluster_centers

    def generate_cluster(self, center):
        """
        Helper function that generates a cluster of cells around 
        the specified center.

        Args:
            center: Center coordinates of the cluster.

        Returns:
            List of cell coordinates in the cluster.
        """
        cluster_size = int(self.random.gauss(4, 1))
        cluster_size = max(cluster_size, 1)

        neighbors = self.grid.get_neighborhood(center, True, True)
        cluster = random.sample(neighbors, min(cluster_size, len(neighbors)))
        return cluster

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

            # Check if the cell is already occupied by a fixed object
            if (x, y) in [agent.pos for agent in self.fixed_cells]:
                continue
            
            # Place an agent based on the density
            if self.random.random() < self.density:

                # Determine the agent's type based on the population fractions
                agent_type = self.random.choices(population=range(self.N),
                                                 weights=self.pop_weights)[0]

                # Determine the agent's initial wealth
                # Note: Lognormal distribution with mean 1.0 and std 1.5 (USA)
                init_wealth = np.random.lognormal(1.0, 1.5)
               
                # Create a new agent
                agent = SchellingAgent((x, y), self, agent_type, init_wealth)
                
                # Add the agent to the grid
                self.grid.place_agent(agent, (x, y))
                
                # Add the agent to the scheduler
                self.schedule.add(agent)

    def calculate_model_stats(self):
        """
        Calculates a number of model statistics, such as:
            - All cluster sizes per population
            - Average cluster size per population
            - Total average cluster size
            - Percolation status per population
            - Percolation status of system
        """

        array = grid2numpy(self)
        self.cluster_sizes_per_pop = find_cluster_sizes_per_pop(self, array)
        self.cluster_summary = cluster_summary(self, self.cluster_sizes_per_pop)
        self.total_avg_cluster_size = np.average([np.mean(self.cluster_sizes_per_pop[value]) if len(self.cluster_sizes_per_pop[value]) > 0 else 0.0 for value in self.cluster_sizes_per_pop.keys()], weights = self.pop_weights)
        self.percolation_per_pop = percolation_detector(self, array)
        self.percolation_system = any([any(self.percolation_per_pop[value]) for value in self.percolation_per_pop.keys()])

    def reset_model_stats(self):
        """
        Resets model statistics.
        """

        self.total_happy = 0
        self.happy_dist = {i: 0 for i in range(self.N)}
        self.happy_t0 = 0
        self.happy_t1 = 0
        self.happy_t2 = 0
        self.happy_t3 = 0
        self.happy_t4 = 0
        self.happy_t5 = 0
        self.happy_t6 = 0
        self.happy_t7 = 0
        self.happy_t8 = 0
        self.happy_t9 = 0

        self.total_wealth = 0
        self.wealth_dist = {i: 0 for i in range(self.N)}
        self.wealth_t0 = 0
        self.wealth_t1 = 0
        self.wealth_t2 = 0
        self.wealth_t3 = 0
        self.wealth_t4 = 0
        self.wealth_t5 = 0
        self.wealth_t6 = 0
        self.wealth_t7 = 0
        self.wealth_t8 = 0
        self.wealth_t9 = 0

    def stopping_condition(self):
        """
        Check if the model should stop running.
        
        Stopping Criterion: Halt if number of happy agents doesn't increase
        from previous step for a number of steps equal to the stopping threshold. 
        """

        # Check if the number of happy agents has increased
        if self.total_happy > self.happy_max:
            # Update previous number of happy agents
            self.happy_max = self.total_happy
            # Reset stopping counter
            self.stopping_cnt = 0
            
        else:
            # Increment stopping counter
            self.stopping_cnt += 1

            # Check if the stopping threshold has been reached
            if self.stopping_cnt >= self.stopping_threshold:
                # Halt the model
                self.running = False

                # If statistics are enabled, calculate the cluster coefficient and half time (computationally expensive)
                if self.stats: 
                    # Segregation coefficient of system
                    self.segregation_coefficient = WeightedAveragepopweights(self, self.cluster_sizes_per_pop)
                    # Wealth segregation coefficient of system
                    self.half_time = CalcHalfTime(self)
        
    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.s
        """

        # Check if the model is running on a server
        if self.server == True:

            # Reset model statistics
            self.reset_model_stats()

            # Advance each agent by one step
            self.schedule.step()

            # Collect data
            self.calculate_model_stats()
            self.datacollector.collect(self)

            # Check stopping condition
            self.stopping_condition()

        # Check if the model is running on a local machine    
        else:
            while self.running == True:

                # Reset model statistics
                self.reset_model_stats()

                # Advance each agent by one step
                self.schedule.step()

                # Collect data
                self.calculate_model_stats()
                self.datacollector.collect(self)

                # Check stopping condition
                self.stopping_condition()
