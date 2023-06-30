import mesa
import random


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
            agent_type: Indicator for the agent's type (minority=1, majority=0)
            init_wealth: Initial wealth of the agent.
        """

        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.wealth = init_wealth

    def calculate_avg_neighbor_wealth(self, neighbors):
        """
        Calculates the average wealth of the agent's neighbors.
        """

        cnt_neighbors = 0
        wealth_neighbors = 0

        # Iterate over all neighbors in moore neighborhood
        for neighbor in self.model.grid.iter_neighbors(self.pos, True):

            # Skip fixed objects
            if neighbor.type == -1:
                continue 

            cnt_neighbors += 1
            wealth_neighbors += neighbor.wealth

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
            # print("step", self.model.schedule.steps)
            # print("self.wealth", self.wealth)
            # print("avg.", avg_neighbor_wealth)

            # Economic rules V2
            # Else update its wealth according to the average of its neighbors
            # print("w/anw", self.wealth / avg_neighbor_wealth)
            # print("a", self.model.alpha)
            # if (self.wealth / avg_neighbor_wealth) <= 1 - self.model.alpha:
            if (1 - self.model.alpha) <= (self.wealth / avg_neighbor_wealth) <= 1:
                self.wealth = 0.5 * self.wealth + 0.5 * avg_neighbor_wealth
                # print("Update")
            else:
                self.wealth = self.wealth
                # print("No Update")

            # Economic rules V1
            # if self.wealth < avg_neighbor_wealth:
            #     self.wealth = 0.5 * self.wealth + 0.5 * avg_neighbor_wealth

        # If an agent has no neighbors, keep its wealth
        else:
            self.wealth = self.wealth
            # print("No neighbors")
        
        # Update the model's total wealth
        self.model.total_wealth += self.wealth

        # Update the wealth distribution
        self.model.wealth_dist[self.type] += self.wealth

        # Update individual wealth per agent type attributes for visualiaztion
        var_key = f'wealth_t{self.type}'
        setattr(self.model, var_key, getattr(self.model, var_key)+self.wealth)

    def is_happy(self):
        """
        Determine whether the agent is happy or not.
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
            self.model.happy += 1
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

    def __init__(self, size=100, density=0.9, fixed_areas_pc=0.0, pop_weights=[0.5, 0.3, 0.2], homophily=4, cluster_threshold=4, alpha=0.5, stopping_threshold=5, server=False):
        """ 
        Initialize the Schelling model.

        Args:
            width, height: The width and height of the grid.
            density: The proportion of the grid to be occupied by agents.
            fixed_areas_pc: The proportion of the grid to be occupied by fixed areas.
            N: The number of agent types.
            pop_weights: The proportion of each agent type in the population.
            homophily: The minimum number of similar neighbors an agent needs to be happy.
            cluster_threshold: The minimum number of agents together to count as a cluster.
            cluster_coefficient: The coefficient that indicates how much segregation occurs.
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

        # Set up model statistics
        self.happy = 0
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
        self.cluster_sizes = {}
        self.cluster_data = {}
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

        self.percolation_data = {}
        self.boolean_percolation = 0
        self.cluster_coefficient = 0.0
        self.half_time = 0

        # Define datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "happy": lambda m: m.happy, # Model-level count of happy agents
                "happy_dist": lambda m: m.happy_dist.copy(), # Model-level count of happy agents
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
                "total_wealth": lambda m: m.total_wealth, # Model-level count of total wealth
                "wealth_dist": lambda m: m.wealth_dist.copy(), # Model-level count of total wealth
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
                "total_avg_cluster_size": lambda m: m.total_avg_cluster_size,  # Model-level total average cluster size
                "cluster_sizes": lambda m: m.cluster_sizes.copy(), # Dictonary of cluster sizes
                "cluster_data": lambda m: m.cluster_data.copy(), # Dictonary of cluster data
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

                "percolation_data": lambda m: m.percolation_data.copy(), # Dictonary of percolation data
                "boolean_percolation": "boolean_percolation", # Model-level boolean value if a cluster percolates
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
        self.happy_prev = 0
        self.server = server

        # Add fixed cells
        self.fixed_cells = []
        if fixed_areas_pc > 0:
            self.add_fixed_cells(fixed_areas_pc)

        # Set up agents
        self.setup_agents()

    def add_fixed_cells(self, fixed_areas_pc):
        """
        Adds fixed cells to the grid by randomly creating clusters of fixed cells and placing them on the grid.

        Args:
            fixed_areas_pc: Proportion of fixed cells in the grid.

        Note: Fixed cells are represented as an agent of type -1.
        """

        num_fixed_cells = int(fixed_areas_pc * self.size * self.size)

        # Generate clusters of fixed cells
        cluster_centers = self.generate_cluster_centers(num_fixed_cells)

        for center in cluster_centers:
            # Generate a cluster of fixed cells around the center
            cluster = self.generate_cluster(center)

            for cell in cluster:
                if cell in [agent.pos for agent in self.fixed_cells]:
                    continue

                # Create a new fixed cell agent
                agent = SchellingAgent(cell, self, -1, 0)

                # Add the agent to the grid
                self.grid.place_agent(agent, cell)
                self.fixed_cells.append(agent)

    def generate_cluster_centers(self, num_centers):
        """
        Generates random cluster centers.

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
        Generates a cluster of cells around the specified center.

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

            # Check if the cell is already occupied by a fixed object
            if (x, y) in [agent.pos for agent in self.fixed_cells]:
                continue
            
            # Place an agent based on the density
            if self.random.random() < self.density:
                # Determine if the agent is a minority
                agent_type = self.random.choices(
                    population=range(self.N),
                    weights=self.pop_weights)[0]

                # Create a new agent
                init_wealth = np.random.lognormal(1.0, 1.5) # Lognormal distribution with mean 3 and std 1 (USA)
                agent = SchellingAgent((x, y), self, agent_type, init_wealth)
                # Add the agent to the grid
                self.grid.place_agent(agent, (x, y))
                # Add the agent to the scheduler
                self.schedule.add(agent)

    def calculate_model_stats(self):
        """
        Calculates the number of clusters, mean cluster size and standard deviation
        for each population group. As well as the total average cluster size and 
        it returns the indivisual cluster sizes.
        """
        array = grid2numpy(self)
        self.cluster_sizes = find_cluster_sizes(self, array)
        self.cluster_data = cluster_summary(self, self.cluster_sizes)
        self.total_avg_cluster_size = np.average([np.mean(self.cluster_sizes[value]) if len(self.cluster_sizes[value]) > 0 else 0.0 for value in self.cluster_sizes.keys()], weights = self.pop_weights)
        self.percolation_data = percolation_detector(self, array)
        self.boolean_percolation = any([any(self.percolation_data[value]) for value in self.percolation_data.keys()])
        self.cluster_coefficient = WeightedAveragepopweights(self, self.cluster_sizes)

    def reset_model_stats(self):
        """
        Resets model statistics.
        """

        self.happy = 0
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

        if self.happy_prev >= self.happy: 
            self.stopping_cnt += 1
            if self.stopping_cnt >= self.stopping_threshold:
                self.running = False
                self.half_time = CalcHalfTime(self)
        else:
            self.stopping_cnt = 0
            self.happy_prev = self.happy
        
    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.s
        """

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
