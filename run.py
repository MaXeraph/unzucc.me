import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt
import pprint as pp


# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)
print(type(D))
# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:])  # 1000 random latents
latents = latents[[420, 1, 2, 3, 4, 5, 6, 7, 8, 455]]  # hand-picked top-10

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

pic = PIL.Image.open("0.jpg")
pix = np.array(pic)
pix = np.rint(((pix / 128.0) - 1)).astype(np.uint8)
print(pix.shape)
pp.pprint(pix)
pix = pix.transpose(2,0,1) # CHW <= HWC

# move rgb to the front for some reason

# switch width for height?
# pix = np.swapaxes(pix, 1,2)
pix = np.reshape(pix, (1,3,1024,1024))

# D.print_layers()
print(D.run(pix))
# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)