import numpy as np

def convertToSize(input, size):
    output = np.zeros(size)
    for row in range(len(output)):
        for col in range(len(output[row])):
            if row > len(input) - 1 or col > len(input[row]) - 1:
                output[row][col] = 0
            else:
                output[row][col] = input[row][col]
    return output