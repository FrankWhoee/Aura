import numpy as np
from aura import aura_loader
import os
import time
import random

def break_aura(path,pieces):
    """
    Breaks an aura file into smaller chunks. Saves chunks to local folders.

    :param path:  A string type of the path to the aura file that is being chunked.
    :param pieces: An integer type of how many pieces should result
    """
    array = aura_loader.read_file(path)
    filepath = "../ChunkedAura" + str(time.time())[5:10]
    print("Saving to " + filepath)
    os.mkdir(filepath)
    l,w,n = array.shape
    print(array.shape)
    chunkSize = int(n/pieces)
    print("Chunking into " + str(chunkSize) + " sized pieces.")
    chunk = np.zeros((l, w, chunkSize), dtype=np.float16)
    for piece in range(pieces):
        print("Chunking piece " + str(piece))
        print("Extracting " + str(chunkSize * piece) + " to " + str(chunkSize * piece + chunkSize))
        for i in range(chunkSize):
            chunk[:,:,i] = array[:,:,i + (chunkSize * piece)]
        f = filepath + "/{" + str(l) + "x" + str(w) + "x" + str(chunk.shape[2]) + "}Chunk" + str(piece) + ".aura"
        print("Saving chunk " + str(piece) + " to " + f + "\n")
        chunk.tofile(f)
    print("----------------- CHUNKING COMPLETE -----------------")


def percentise_aura(path,percent):
    """
    Breaks an aura file into two pieces of percent sizes.

    :param path: A string type of the path to the aura file that is being chunked.
    :param percent: A float or double type of the percentage that should be in the first chunk.
    """
    array = aura_loader.read_file(path).T
    random.shuffle(array)
    filepath = "../ChunkedAura" + str(time.time())[5:10]
    print("Saving to " + filepath)
    os.mkdir(filepath)
    n,l,w = array.shape
    print(array.shape)
    print("Chunking into " + str(percent * 100) +"% and " + str((1-percent) * 100) + "%")
    size1 = int(n * percent)
    size2 = int(n * (1-percent))

    print("Chunk1 size = " + str(size1))
    print("Chunk2 size = " + str(size2))

    chunk1 = np.zeros((l, w, size1), dtype=np.float16)
    chunk2 = np.zeros((l, w, size2), dtype=np.float16)

    print("Chunking piece 1")
    for i in range(size1):
        chunk1[:,:,i] = array[i]
    f1 = filepath + "/{" + str(chunk1.shape[0]) + "x" + str(chunk1.shape[1]) + "x" + str(chunk1.shape[2]) + "}Chunk1.aura"
    print("Saving chunk1 to " + f1 + "\n")
    chunk1.tofile(f1)

    for i in range(size2):
        chunk2[:,:,i] = array[i + (size1)]
    f2 = filepath + "/{" + str(chunk2.shape[0]) + "x" + str(chunk2.shape[1]) + "x" + str(chunk2.shape[2]) + "}Chunk2.aura"
    print("Saving chunk1 to " + f2 + "\n")
    chunk2.tofile(f2)

    print("----------------- CHUNKING COMPLETE -----------------")

