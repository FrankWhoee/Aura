from keras.models import load_model
from keras.models import Model
from keras.models import save_model
from keras.losses import categorical_crossentropy
import keras

def generate_encapsulate_model_with_output_layer_names(model, output_layer_names):
    enc_model = Model(
        inputs=model.input,
        outputs=list(map(lambda oln: model.get_layer(oln).output, output_layer_names))
    )
    return enc_model


model = load_model("Model-11.hf")
model.summary()
for layer in model.layers:
     print(layer.name)

output_layer_names = "conv2d_1 max_pooling2d_1 conv2d_2 max_pooling2d_2 conv2d_3 max_pooling2d_3 conv2d_4 flatten_1 dense_1 dropout_1 dense_2 dense_3 dropout_2 dense_4 dropout_3 dense_5 dense_6 dropout_4 dense_7 dense_8 dense_9".split(" ")


enc_model = generate_encapsulate_model_with_output_layer_names(model, output_layer_names)

# enc_model.compile(optimizer=keras.optimizers.Adadelta(),loss=categorical_crossentropy,metrics=['accuracy'])
save_model(enc_model, "enc_model.h5")