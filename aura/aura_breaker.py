import numpy as np
from aura import aura_loader
import os
import time

def break_aura(path,pieces):
    array = aura_loader.read_file(path)
    filepath = "../ChunkedAura" + str(time.time())[5:10]
    print("Saving to " + filepath)
    os.mkdir(filepath)
    l,w,n = array.shape
    print(array.shape)
    chunkSize = int(n/pieces)
    print("Chunking into " + str(chunkSize) + " sized pieces.")
    chunk = np.zeros((l, w, chunkSize))
    for piece in range(pieces):
        print("Chunking piece " + str(piece))
        print("Extracting " + str(chunkSize * piece) + " to " + str(chunkSize * piece + chunkSize))
        for i in range(chunkSize):
            chunk[:,:,i] = array[:,:,i + (chunkSize * piece)]
        f = filepath + "/{" + str(l) + "x" + str(w) + "x" + str(chunk.shape[2]) + "}Chunk" + str(piece) + ".aura"
        print("Saving chunk " + str(piece) + " to " + f + "\n")
        chunk.tofile(f)
    print("----------------- CHUNKING COMPLETE -----------------")

break_aura("../Dataset/{136x136x2353}HealthyTrainset.aura", 13)