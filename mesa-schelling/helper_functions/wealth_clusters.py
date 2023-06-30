import numpy as np

def WealthOnGrid(self):
    """
    Function that takes the model as input and finds the wealth of the agents at different points in the grid and puts that 
    into a numpy array. The wealth is then also plotted in a heatmap and returns the array with the wealths.
    """

    # Create a numpy array of the grid size
    wealth_on_grid = np.zeros((self.grid.width, self.grid.height))

    # Loop through each cell and check if it contains an agent
    for cell in self.grid.coord_iter():
        cell_content, x, y = cell

        # If the cell contains an agent, add the agent type to the array
        if cell_content:
            wealth_on_grid[x][y] = cell_content.wealth
    
    return wealth_on_grid

def WealthSegregation(self, x, y):
    """
    Function that takes as input the model, the array with wealths per grid location and an x and y integer. This function 
    calculates the variance of a growing square in the grid, with starting location (x,y). It returns an array with the variance
    per square side size.
    """

    # Create a numpy array of the grid size
    wealth_on_grid = WealthOnGrid(self)

    # Determine the size of the grid
    N = self.grid.width

    # Create empty lists for the variance and the square size
    var_list = []
    L_list = []

    # Loop over all possible square sizes
    for L in range(1, N):
        
        # Determine the coordinates of the square based on the initial coordinate and L
        start_row = x - (L // 2)
        end_row = start_row + L
        start_col = y - (L // 2)
        end_col = start_col + L

        # Adjust the coordinates if they exceed the array boundaries
        if end_row > N:
            start_row -= end_row - N
            end_row = N
        if end_col > N:
            start_col -= end_col - N
            end_col = N
        if start_row < 0:
            end_row -= start_row
            start_row = 0
        if start_col < 0:
            end_col -= start_col
            start_col = 0

        # Calculate the variance of wealth in the square
        var = np.var(wealth_on_grid[start_row:end_row, start_col:end_col])

        # Append variance to list
        var_list.append(var)
        L_list.append(L)
        
    # Delete the first element of both lists (zero variance for square of 1x1)
    var_list.pop(0)
    L_list.pop(0)    
    
    return var_list

def WealthSegregationAverage(self):
    """
    Fuction that finds the average variance for different square sizes L over all possible starting coordinates (x,y).
    """

    # Create an empty array for the variance
    var = np.zeros(self.grid.width-2)

    # Fiding the variances for all starting coordinates and summing them together
    for x in range(0, self.grid.width):
        for y in range(0, self.grid.height):
            var_list = WealthSegregation(self, x, y)
            var += var_list

    # Calculating the average
    average_vars = var/self.grid.width
    L = np.array(range(1, self.grid.width-1))

    return average_vars, L

def CalcHalfTime(self):
    """
    Function that calculates the half time of the wealth segregation. It does this by interpolating the half value of the
    variance in the average variance array.
    """

    # Finding the average variance and the square size
    average_vars, L = WealthSegregationAverage(self)

    # Finding the half time
    end_value = average_vars[len(average_vars)-1] # The last value of the variance
    half_time = np.interp(0.5 * end_value, average_vars, L) # Interpolating the half time
    
    return half_time