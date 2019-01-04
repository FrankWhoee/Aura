import numpy as np

def convertToSize(input, size):
    output = np.zeros(size)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            if row < len(input) and col < len(input[row]):
                output[row][col] = input[row][col]

    return output