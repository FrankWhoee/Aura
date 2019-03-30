from __future__ import print_function
from keras.models import load_model
from aura.aura_loader import get_data

model = load_model("Model-11.hf")

root = "../Aura_Data/Chunked/"
test_paths = [root + "ChunkedHealthy/{136x136x2353}Chunk0.aura", root + "ChunkedRIDER/{256x256x3511}Chunk0.aura",
              root + "ChunkedSmallerCPTAC/{256x256x543}Chunk0.aura"]
test_data, test_label = get_data(test_paths)
test_n, test_l, test_w = test_data.shape
y_test = test_label.copy()
x_test = test_data.reshape(test_n, test_l, test_w, 1)
print(x_test.shape)
print(y_test.shape)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
