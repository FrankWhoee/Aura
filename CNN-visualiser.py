# from keras.utils import plot_model
# from keras.models import load_model
# print("Loading model...")
# model = load_model("Model-11.hf")
# print("Model loaded. Plotting model...")
# plot_model(model, to_file='model.png', show_shapes=True)
# print("Plotting complete. File is ready at model.png")

from keras.utils import plot_model
from keras.models import load_model
from vis.visualization.saliency import visualize_saliency
from vis.visualization.activation_maximization import visualize_activation
from matplotlib import pyplot
import numpy
from scipy.misc import imresize
from scipy.misc import imshow
from aura.aura_loader import read_file

print("Loading images...")
root = "../Aura_Data/Dataset/"
cancer_path = "../Aura_Data/Chunked/ChunkedCPTAC/{256x256x10861}Chunk0.aura"
healthy_path = root + "{136x136x22118}HealthyTestset.aura"
# imgh = read_file(healthy_path).T[5]
# imageHealthy = imresize(imgh,(256,256)).reshape(1,256,256,1)
# imshow(imgh)
imgc = read_file(cancer_path).T[10]
imageCancer = imgc.reshape(1,256,256,1)
imshow(imgc)
print("Image loaded.")

print("Loading model...")
model = load_model("Model-11.hf")
print("Model loaded.")
print("Visualising filters...")
layers = visualize_saliency(model=model,layer_idx=10, filter_indices=None, seed_input=imageCancer)
print(layers.shape)
print("Filters visualised.")
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.pcolormesh(layers[:,:,2])
pyplot.show()

# print("Loading model...")
# model = load_model("Model-11.hf")
# print("Model loaded.")
# print("Visualising filters...")
# layers = visualize_activation(model,layer_idx=13)
# print(layers.shape)
# print("Filters visualised.")
# pyplot.figure(dpi=300)
# pyplot.axes().set_aspect('equal', 'datalim')
# # pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(layers[:,:,0])
# pyplot.show()