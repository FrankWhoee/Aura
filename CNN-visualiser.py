from keras.utils import plot_model
from keras.models import load_model
from IPython.display import SVG
# from keras.utils import mo
print("Loading model...")
model = load_model("Model-10-2.hdf5")
print("Model loaded. Plotting model...")
plot_model(model, to_file='model.png', show_shapes=True)
print("Plotting complete. File is ready at model.png")
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
