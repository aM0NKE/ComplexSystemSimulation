import numpy as np
import scipy as sp

def WeightedAveragepopweights(self, cluster_sizes):
    S = 0
    
    for i in cluster_sizes:
        a = 0
        for j in range(len(cluster_sizes[i])):
            a += cluster_sizes[i][j]**2
        
        S += (1/(self.grid.width*self.grid.height*self.pop_weights[i])**2)*a
    
    return S/self.N 

def grid2numpy(self):
    """
    This function converts the grid to a numpy array.
    """

    array = -1 * np.ones((self.grid.width, self.grid.height), dtype=int)

    for cell in self.grid.coord_iter():
        _, x, y = cell
        if len(self.grid.get_cell_list_contents((x, y))) != 0:
            agent = self.grid.get_cell_list_contents((x, y))[0]  # Assuming only one agent per cell
            array[x, y] = agent.type
    return array

def cluster_finder(self, mask):
    """
    This helper function has as imput a binary matrix of one population group
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
    return clusters[clusters >= self.cluster_threshold]

def find_cluster_sizes(self, array):
    """
    This function finds all the cluster size(s) for all the populations on the 2D grid.

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
            cluster_sizes[value] = cluster_finder(self, mask)
    return cluster_sizes

def cluster_summary(self, cluster_sizes):
    """
    This function calculates the number of clusters, mean cluster size 
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
        if len(cluster_sizes[value]) != 0:
            cluster_data[value] = [len(cluster_sizes[value]), np.mean(cluster_sizes[value]),
                                    np.std(cluster_sizes[value])]
        else:
            cluster_data[value] = [0, 0, 0]

        # Update individual cluster sizes per agent type attributes for visualiaztion
        var_key = f'cluster_t{value}'
        setattr(self, var_key, cluster_data[value][1])
    return cluster_data

