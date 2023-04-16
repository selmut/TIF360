import deeptrack as dt
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

particle = dt.Sphere(position=lambda: (np.random.uniform(20, 40), np.random.uniform(20, 40)))
optics = dt.Fluorescence(output_region=(0, 0, 64, 64))
sample = optics(particle) >> dt.Gaussian(sigma=0.1)
im = sample.update()()
plt.imshow(im[..., 0])

Training_dataset = [sample.update()() for i in range(1000)]
Validation_dataset = [sample.update()() for i in range(100)]

print(Training_dataset)


def label_function(image):
    position = image.get_property("position")
    return np.array(position)


Training_labels = [label_function(Training_dataset[i]) for i in range(len(Training_dataset))]
Validation_labels = [label_function(Validation_dataset[i]) for i in range(len(Validation_dataset))]

ae = Autoencoder((64, 64, 1))
model = ae.create_model()


h = model.fit(x=np.array(Training_dataset), y=np.array(Training_labels), validation_data=(np.array(Validation_dataset),
              np.array(Validation_labels)), epochs=40)
p = model.predict(np.array(Validation_dataset))
