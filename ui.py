import tkinter as tk
from PIL import ImageTk, Image
import gan_model # import the GAN model code

# Create a Tkinter window
window = tk.Tk()
window.title("Anime Image Generator")

# Add a label for the input image
input_label = tk.Label(window, text="Input Image")
input_label.pack()

# Add a canvas for displaying the input image
input_canvas = tk.Canvas(window, width=256, height=256)
input_canvas.pack()

# Add a label for the output image
output_label = tk.Label(window, text="Output Image")
output_label.pack()

# Add a canvas for displaying the output image
output_canvas = tk.Canvas(window, width=256, height=256)
output_canvas.pack()

# Define a function to generate anime-style images
def generate_anime_image():
    # Load the input image
    input_image = Image.open(input_image_path)
    input_image = input_image.convert("RGB")
    
    # Resize and normalize the input image
    input_image = input_image.resize((64, 64))
    input_image = gan_model.normalize_input(input_image)
    
    # Generate the output image
    output_image = gan_model.generate_image(input_image)
    
    # Resize and denormalize the output image
    output_image = output_image.resize((256, 256))
    output_image = gan_model.denormalize_output(output_image)
    
    # Display the input and output images on the canvas
    input_canvas.image = ImageTk.PhotoImage(input_image)
    input_canvas.create_image(0, 0, anchor=tk.NW, image=input_canvas.image)
    
    output_canvas.image = ImageTk.PhotoImage(output_image)
    output_canvas.create_image(0, 0, anchor=tk.NW, image=output_canvas.image)

# Add a button to generate anime-style images
generate_button = tk.Button(window, text="Generate Anime Image", command=generate_anime_image)
generate_button.pack()

# Start the Tkinter event loop
window.mainloop()
