import mesa
import random
import numpy as np
import scipy as sp

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
        if self.type != -1: # Don't move fixed objects
            self.move()


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, width=100, height=100, density=0.8, fixed_areas_pc=.1, N=4, pop_weights=[0.6, 0.2, 0.1, 0.1], homophily=3):
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
        self.cluster_sizes = {}
        self.cluster_data = {}
        self.total_cluster_average = 0.0

        self.datacollector = mesa.DataCollector(
            {"happy": "happy"},  # Model-level count of happy agents
            # {"total_cluster_average": "total_cluster_average"},  # Model-level total average cluster size
            # {"cluster_sizes": "cluster_sizes"}, # Dictonary of cluster sizes
            # {"cluster_data": "cluster_data"}, # Dictonary of cluster data
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
        )
        self.datacollector.collect(self)

        # Variable to halt model
        self.running = True

        # Add fixed cells
        self.fixed_cells = []
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

        num_fixed_cells = int(fixed_areas_pc * self.width * self.height)

        # Generate clusters of fixed cells
        cluster_centers = self.generate_cluster_centers(num_fixed_cells)

        for center in cluster_centers:
            # Generate a cluster of fixed cells around the center
            cluster = self.generate_cluster(center)

            for cell in cluster:
                if cell in [agent.pos for agent in self.fixed_cells]:
                    continue

                # Create a new fixed cell agent
                agent = SchellingAgent(cell, self, -1)

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
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
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
                    weights=self.pop_weights,)[0]

                # Create a new agent
                agent = SchellingAgent((x, y), self, agent_type)
                # Add the agent to the grid
                self.grid.place_agent(agent, (x, y))
                # Add the agent to the scheduler
                self.schedule.add(agent)

    def mesa_grid_to_numpy_grid(self):
        # Convert the grid to a NumPy array
        array = np.zeros((self.grid.width, self.grid.height), dtype=int)
        for cell in self.grid.coord_iter():
            _, x, y = cell
            if len(self.grid.get_cell_list_contents((x, y))) != 0:
                agent = self.grid.get_cell_list_contents((x, y))[0]  # Assuming only one agent per cell
                array[x, y] = agent.type
        return array
    
    def cluster_finder(self, mask):
        """This function has as imput a binary matrix of one population group
        and returns the size of cluster(s).

        Args:
            mask (2D numpy array): Matrix of intergers where 0 is not part of a cluster
            and 1 is part of a cluster.

        Returns:
            cluster: an array of clusters sizes for a input population group (mask)
        """
        # Labels the clusters
        lw, _ = sp.ndimage.label(mask)

        # sums the agents that are part of a cluster
        clusters = sp.ndimage.sum(mask, lw, index=np.arange(lw.max() + 1))
        return clusters[1:]

    def find_cluster_sizes(self, array):
        """This function finds all the cluster size(s) for all the populations on the 2D grid.

        Args:
            array (2D numpy array): Matrix of intergers where 0 is the empty space
                                    and the other intergers a population agent.

        Returns:
            cluster_sizes (dictonary): The keys are the population value (population group)
                                    and the values is an array of cluster size(s).
        """
        unique_values = np.unique(array)
        cluster_sizes = {}

        for value in unique_values:
            # value 0 is an empty space thus not part of the cluster.
            if value >= 0:
                # Isolate the selected population group form the rest (makt it a binary matrix)
                mask = array == value
                # find the cluster size(s) for the selected population group.
                cluster_sizes[value] = self.cluster_finder(mask)
        return cluster_sizes

    def cluster_analysis(self, cluster_sizes):
        """This function calculates the number of clusters, mean cluster size 
        with standard deviation.

        Args:
            cluster_sizes (dictonary): The keys are the population value (population group)
                                    and the values is an array of cluster size(s).

        Returns:
            cluster_data (dictonary): The keys are the population value (population group)
                                    and the values is an array of number of clusters,
                                    mean cluster size and standard deviation.
        """
        cluster_data = {}
        for value in cluster_sizes.keys():
            cluster_data[value] = [len(cluster_sizes[value]), np.mean(cluster_sizes[value]),
                                    np.std(cluster_sizes[value])]
        return cluster_data
    
    def average_cluster_size_system(self, cluster_sizes):
        cluster_average = 0
        for value in cluster_sizes.keys():
            cluster_average += np.mean(cluster_sizes[value])
        return cluster_average / len(cluster_sizes)
    
    def calculate_cluster_sizes(self):
        array = self.mesa_grid_to_numpy_grid()
        self.cluster_sizes = self.find_cluster_sizes(array)
        self.cluster_data = self.cluster_analysis(self.cluster_sizes)
        self.total_cluster_average = self.average_cluster_size_system(self.cluster_sizes)
        
    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """

        # Reset counter of happy agents
        self.happy = 0
        
        # Advance each agent by one step
        self.schedule.step()

        # Collect data
        self.calculate_cluster_sizes()
        self.datacollector.collect(self)

        # Halt if no unhappy agents
        if self.happy == self.schedule.get_agent_count():
            self.running = False