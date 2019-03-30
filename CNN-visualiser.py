from keras.utils import plot_model
from keras.models import load_model
from vis.visualization.saliency import visualize_saliency
from vis.visualization.activation_maximization import visualize_activation
from matplotlib import pyplot


def generate_model_image(model_path):
    """
    Generates a flow chart of the model to model.png
    :param model_path: Path of the Keras model to be loaded. Expects string input.
    :return: None
    """
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded. Plotting model...")
    plot_model(model, to_file='model.png', show_shapes=True)
    print("Plotting complete. File is ready at model.png")


def visualize_saliency_map(model_path, layer, image, sensitivity):
    """
    Visualises saliency maps for the specified image.
    :param model_path: Path of the Keras model to be loaded. Expects string input.
    :param layer: Which layer of the neural network is to be visualised. Expects integer input.
    :param image: Image to be analyzed by the neural network to create saliency map. Expects 2D matrix input.
    :param sensitivity: Saliency sensitivity. Goes from 0 to 3. Expects integer input.
    :return: None
    """
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.")
    print("Visualising saliency map...")
    layers = visualize_saliency(model=model, layer_idx=layer, filter_indices=None, seed_input=image)
    print(layers.shape)
    print("Filters visualised.")
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.pcolormesh(layers[:, :, sensitivity])
    pyplot.show()


def visualize_feature_maps(model_path, layer):
    """
    Visualises the feature maps within a Keras model.
    :param model_path: Path of the Keras model to be loaded. Expects string input.
    :param layer: Which layer of the model to extract a feature map of. Expects integer input.
    :return: None
    """
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.")
    print("Visualising filters...")
    layers = visualize_activation(model, layer_idx=layer)
    print(layers.shape)
    print("Filters visualised.")
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.pcolormesh(layers[:, :, 0])
    pyplot.show()
