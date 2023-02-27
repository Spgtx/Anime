from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load generator model
generator = load_model('anime_generator.h5')

# Load real image
img = load_img('real_image.jpg', target_size=(64, 64))
img_array = img_to_array(img)
img_array = (img_array - 127.5) / 127.5 # Rescale pixel values to [-1, 1]
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

# Generate anime-style image
generated_img_array = generator.predict(img_array)
generated_img_array = 0.5 * generated_img_array + 0.5 # Rescale pixel values to [0, 1]
generated_img = generated_img_array[0].astype('uint8')

# Save generated image
generated_img = Image.fromarray(generated_img)
generated_img.save('generated_image.jpg')
