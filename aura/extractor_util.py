import numpy as np

def convertToSize(input, size):
    output = np.zeros(size)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            if row < len(input) and col < len(input[row]):
                output[row][col] = input[row][col]

    return output


def reshape(input,size):
    output = np.zeros(size)
    for image in range(output.shape[2]):
        output[:,:,image] = convertToSize(input[:,:,image],size[0:2])
    return output