from __future__ import print_function
import numpy as np
from keras.models import load_model
from aura.extractor_util import reshape
from matplotlib import pyplot as plt
from aura.extractor_util import parseAuraDimensions as pAD
from aura.aura_loader import read_file

root = "../Aura_Data/";
cancerPath = root + "ChunkedCancerTestset/"
healthyPath = root + "ChunkedHealthyTestset/"
cancerSize = "{256x256x270}"
healthySize = "{136x136x181}"

cl,cw,cn = pAD(cancerSize)
hl,hw,hn = pAD(healthySize)
fl, fw = max(cl, cw, hl, hw), max(cl, cw, hl, hw)
fn = cn + hn
num_classes = 2

model = load_model("Model-10-2.hdf5")

cancerous_test_data = read_file(path=cancerPath + cancerSize + "Chunk12.aura").T
healthy_test_data = read_file(path=healthyPath + healthySize + "Chunk12.aura")
healthy_test_data = reshape(healthy_test_data, (fl,fw, hn)).T
test_data = np.zeros((fn, fl,fw))
for i in range(cn):
    test_data[i] = cancerous_test_data[i]
for i in range(hn):
    test_data[i + cn] = healthy_test_data[i]

x_test = test_data

x_test = test_data.reshape(fn,fl,fw,1)

image = test_data[200]

plt.imshow(image)
plt.show()

print(model.predict(image.reshape(1,256,256,1)))