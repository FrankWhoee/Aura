from keras.utils import plot_model
from keras.models import load_model
print("Loading model...")
model = load_model("Model-11.hf")
print("Model loaded. Plotting model...")
plot_model(model, to_file='model.png', show_shapes=True)
print("Plotting complete. File is ready at model.png")