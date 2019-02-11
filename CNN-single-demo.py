from __future__ import print_function
from aura.aura_loader import read_file
from keras.models import load_model
from aura.decode import decode
from aura.decode import preprocess
from aura.decode import view_image as view


root = "../Aura_Data/Dataset/";

model = load_model("Model-11-1.hf")
# image = read_file(root + "ChunkedHealthyTestset/{136x136x181}Chunk1.aura").T[50]
imageCancer = read_file(root + "{256x256x7021}RIDERTestset.aura").T[int(input("Choose image from cancerous test set (0-7020)"))]
imageHealthy = read_file(root + "{136x136x22118}HealthyTestset.aura").T[int(input("Choose image from healthy test set (0-22117)"))]
imageBTP = read_file(root + "{256x256x879}BTPTestset.aura").T[int(input("Choose image from a cancerous test set from another database (0-879)"))]
# image = dcm.read_file(root + "Unextracted/CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/8-AX 3D SPGR-43615/000199.dcm").pixel_array

print("")

all_images = [imageHealthy, imageCancer, imageBTP]
all_predictions = []

for index, image in enumerate(all_images):
    view(image)
    all_images[index] = preprocess(image)

for index, image in enumerate(all_images):
    all_predictions.append(decode(model.predict(image)))

for i,prediction in enumerate(all_predictions):
    if prediction[0][1] > 0.5:
        print("Patient "+str(i)+" is healthy.")
        print("Confidence: " + str(prediction[0][1] * 100)[0:4] + "%\n")
    elif prediction[1][1] > 0.5:
        print("Patient " + str(i) + " has GBM.")
        print("Confidence: " + str(prediction[1][1] * 100)[0:4] + "%\n")
    elif prediction[2][1] > 0.5:
        print("Patient " + str(i) + " has GBM.")
        print("Confidence: " + str(prediction[2][1] * 100)[0:4] + "%\n")
