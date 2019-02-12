from __future__ import print_function
from keras.models import load_model
from aura.aura_loader import get_data

model = load_model("Model-11.hf")

# Prepare paths for GCP training
root = "../Aura_Data/"
test_paths = [root + "{136x136x22118}HealthyTestset.aura", root + "{256x256x7021}RIDERTestset.aura",
              root + "{256x256x879}BTPTestset.aura"]
test_data, test_label = get_data(test_paths)
test_n, test_l, test_w = test_data.shape
x_test = test_data.reshape(test_n, test_l, test_w, 1)
y_test = test_label.copy()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
