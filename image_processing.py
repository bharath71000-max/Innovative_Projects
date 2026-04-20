from PIL import Image, ImageFilter

def main():
    # Open image
    img = Image.open("input.jpg")

    # Convert to grayscale
    gray = img.convert("L")
    gray.save("gray.jpg")

    # Apply blur filter
    blur = img.filter(ImageFilter.BLUR)
    blur.save("blur.jpg")

    # Find edges
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges.save("edges.jpg")

    print("Processing Completed!")

# Call main function
if __name__ == "__main__":
    main()