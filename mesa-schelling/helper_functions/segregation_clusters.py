import numpy as np
import scipy as sp

def WeightedAveragepopweights(self, cluster_sizes_per_pop):
    """
    This function calculates the weighted average of the cluster sizes.
    Or segregation coefficient S.

    Args:
        cluster_sizes_per_pop:  A dictionary where the keys are the population group and the values
                        are an array of cluster sizes for that population group.
    """

    # Define the segregation coefficient
    S = 0
    
    # Loop through each population group
    for i in cluster_sizes_per_pop:
        
        # Calculate the sum of the squares of the cluster sizes
        a = 0
        for j in range(len(cluster_sizes_per_pop[i])):
            a += cluster_sizes_per_pop[i][j]**2
        
        # Calculate the weighted average of the cluster sizes
        S += (1 / (self.grid.width * self.grid.height * self.pop_weights[i])**2) * a
    
    return S / self.N

def grid2numpy(self):
    """
    This function converts the grid to a numpy array.
    """

    # Create a numpy array of the grid size
    array = -1 * np.ones((self.grid.width, self.grid.height), dtype=int)

    # Loop through each cell and check if it contains an agent
    for cell in self.grid.coord_iter():
        _, x, y = cell

        # If the cell contains an agent, add the agent type to the array
        if len(self.grid.get_cell_list_contents((x, y))) != 0:
            agent = self.grid.get_cell_list_contents((x, y))[0]  # Assuming only one agent per cell
            array[x, y] = agent.type
    
    return array

def cluster_finder(self, mask):
    """
    This helper function has as imput a binary matrix of one population group
    and returns the size of cluster(s).

    Args:
        mask: Matrix of intergers where 0 is not part of a cluster and 1 is 
              part of a cluster.

    Returns:
        cluster: an array of clusters sizes for a input population group (mask)
    """

    # Labels the clusters
    lw, _ = sp.ndimage.label(mask)
    
    # Sums the agents that are part of a cluster
    clusters = sp.ndimage.sum(mask, lw, index=np.arange(lw.max() + 1))

    # Returns the cluster sizes
    return clusters[clusters >= self.cluster_threshold]

def find_cluster_sizes_per_pop(self, array):
    """
    This function finds all the cluster size(s) for all the populations on the 2D grid.

    Args:
        array: Matrix of intergers where 0 is the empty space
               and the other intergers a population agent.

    Returns:
        cluster_sizes_per_pop: The keys are the population value (population group)
                               and the values is an array of cluster size(s).
    """

    # Find the unique population values
    unique_values = np.unique(array)

    # Create a dictionary to store the cluster sizes per population group
    cluster_sizes_per_pop = {}

    # Loop through each population group
    for value in unique_values:

        # Check if the value is not an fixed object
        if value >= 0:
            # Isolate the selected population group form the rest (makt it a binary matrix)
            mask = array == value
            # Find the cluster size(s) for the selected population group.
            cluster_sizes_per_pop[value] = cluster_finder(self, mask)

    return cluster_sizes_per_pop

def cluster_summary(self, cluster_sizes_per_pop):
    """
    This function calculates the number of clusters, mean cluster size 
    with standard deviation.

    Args:
        cluster_sizes_per_pop: The keys are the population value (population group)
                               and the values is an array of cluster size(s).

    Returns:
        cluster_data: The keys are the population value (population group)
                      and the values is an array of number of clusters,
                      mean cluster size and standard deviation.
    """

    # Create a dictionary to store the cluster data
    cluster_data = {}

    # Loop through each population group
    for value in cluster_sizes_per_pop.keys():

        # Calculate the number of clusters, mean cluster size and standard deviation
        if len(cluster_sizes_per_pop[value]) != 0:
            cluster_data[value] = [len(cluster_sizes_per_pop[value]), np.mean(cluster_sizes_per_pop[value]),
                                    np.std(cluster_sizes_per_pop[value])]
        else:
            cluster_data[value] = [0, 0, 0]

        # Update individual cluster sizes per agent type attributes for visualiaztion
        var_key = f'cluster_t{value}'
        setattr(self, var_key, cluster_data[value][1])
        
    return cluster_data

