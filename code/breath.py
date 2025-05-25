from PIL import Image

# Define image size and color (white background)
width, height = 800, 600
background_color = (255, 255, 255)  # RGB for white

# Create a new image
img = Image.new("RGB", (width, height), color=background_color)

# Save the image
img.save("output.png")

print("Empty image 'output.png' created successfully.")
