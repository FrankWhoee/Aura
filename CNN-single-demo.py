from __future__ import print_function
from aura.aura_loader import read_file
from keras.models import load_model
from aura.decode import decode
from aura.decode import preprocess
from aura.decode import view_image as view
from aura.aura_loader import parse_aura_dimensions
from sys import stderr
from time import sleep

print("Loading model...")
model = load_model("Model-11.hf")
print("Model loaded.")

# Prepare paths
root = "../Aura_Data/Dataset/"
cancer_path = "../Aura_Data/Chunked/ChunkedCPTAC/{256x256x10861}Chunk0.aura"
healthy_path = root + "{136x136x22118}HealthyTestset.aura"
btp_path = root + "{256x256x879}BTPTestset.aura"

cl, cw, cn = parse_aura_dimensions(cancer_path)
hl, hw, hn = parse_aura_dimensions(healthy_path)
bl, bw, bn = parse_aura_dimensions(btp_path)


def query_user(question, n, min=0):
    """
    Queries a user from the console, and returns the user's

    :param question: Type string that is asked to the user.
    :param n: Upper bound
    :param min: Lower bound
    :return: Integer type
    """

    user_question = question + " (" + str(min) + "-" + str(n) + "): "
    image_index = input(user_question)
    while not image_index.isdigit() or int(image_index) > n or int(image_index) < 0:
        stderr.write("\nPlease enter a number between " + str(min) + " and " + str(n) + "\n")
        sleep(0.01)
        image_index = input(user_question)
    return int(image_index)


def get_most_confident_prediction(prediction):
    highest_label, highest_confidence = "", 0
    for item in prediction:
        if item[1] > highest_confidence:
            highest_label = item[0]
            highest_confidence = item[1]
    return highest_label, highest_confidence


# Query users for input
# cancer_image_index = query_user("Choose image from cancerous test set", cn - 1)
# healthy_image_index = query_user("Choose image from healthy test set", hn - 1)
# btp_image_index = query_user("Choose image from another cancerous test set", bn - 1)

cancer_image_index = 7
healthy_image_index = 529
btp_image_index = 100

imageHealthy = read_file(healthy_path).T[healthy_image_index]
imageCancer = read_file(cancer_path).T[cancer_image_index]
imageBTP = read_file(btp_path).T[btp_image_index]

print("Processing images...")
# Compile images into one array
all_images = [imageHealthy, imageCancer, imageBTP]
all_predictions = []

# Preprocess all images and plot them.
for index, image in enumerate(all_images):
    view(image)
    all_images[index] = preprocess(image)
print("Images processed.")

print("Analysing images...")
# Use model to predict all images, and compile into all_predictions
for index, image in enumerate(all_images):
    all_predictions.append(decode(model.predict(image)))
print("Images analysed. Processing results...")
print("\n---------------------RESULTS---------------------")
# Print out results.
for i, prediction in enumerate(all_predictions):
    print("Patient " + str(i) + " is/has " + get_most_confident_prediction(prediction)[0])
    print(get_most_confident_prediction(prediction)[1])
