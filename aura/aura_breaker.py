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

def percentise_aura(path,percent):
    array = aura_loader.read_file(path)
    filepath = "../ChunkedAura" + str(time.time())[5:10]
    print("Saving to " + filepath)
    os.mkdir(filepath)
    l,w,n = array.shape
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
        chunk1[:,:,i] = array[:,:,i]
    f1 = filepath + "/{" + str(l) + "x" + str(w) + "x" + str(size1.shape[2]) + "}Chunk1.aura"
    print("Saving chunk1 to " + f1 + "\n")
    chunk1.tofile(f1)

    for i in range(size2):
        chunk2[:,:,i] = array[:,:,i + (size1)]
    f2 = filepath + "/{" + str(l) + "x" + str(w) + "x" + str(size2.shape[2]) + "}Chunk1.aura"
    print("Saving chunk1 to " + f2 + "\n")
    chunk2.tofile(f2)

    print("----------------- CHUNKING COMPLETE -----------------")

percentise_aura("../../Aura_Data/Unchunked/{136x136x221182}Healthy.aura", 0.99)