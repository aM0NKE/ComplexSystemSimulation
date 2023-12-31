import numpy as np
import scipy as sp

def percolation_detector(self, array):
    """
    This function checks if a population group contains a cluster that percolates,
    either vertically, horizontally or both.

    Args:
        array:  Matrix of intergers where 0 is the empty space and the other 
                intergers a population agent.

    Returns:
        percolation_check:  The keys are the population value (population group) and
                            the values is an array of boolian values. The first 
                            vertical percolation and the second horizontal.
    """

    # Get the number of populations
    unique_values = np.unique(array)

    # Create a dictionary to store the percolation check for each population group
    percolation_check = {}

    # Loop through each population group and determines the cluster(s) and their sizes.
    for value in unique_values:
        
        # Check if agent type is not a fixed object
        if value >= 0:

            # Create a mask of the population group
            mask = array == value

            # Extract all the clusters of the population group
            labels, num_clusters = sp.ndimage.label(mask)
            clusters = sp.ndimage.sum(mask, labels, index=np.arange(labels.max() + 1))
            
            # Set the percolation checks to false
            percolates_vertically = False
            percolates_horizontally = False

            # Loop through each cluster of a single poplulation group and check if it percolates
            # If the cluster is not big enough to percolate or if a previous cluster already percolated the check is skipped.
            for label in range(1,num_clusters+1):

                # Check if the cluster is big enough to percolate vertically
                if percolates_vertically == False and clusters[label] >= mask.shape[0]:
                    if label in labels[:,0] and label in labels[:,-1]:
                        percolates_vertically = True
                    
                # Check if the cluster is big enough to percolate horizontally
                if percolates_horizontally == False  and clusters[label] >= mask.shape[1]:
                    if label in labels[0,:] and label in labels[-1,:]:
                        percolates_horizontally = True

            # Store the percolation check for the population group
            percolation_check[value] = [percolates_vertically, percolates_horizontally]

    return percolation_check