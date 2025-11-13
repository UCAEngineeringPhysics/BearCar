from PIL import Image
from pathlib import Path
import os

# -----------------------------------------------------------------
# --- 1. EDIT THIS LINE ---
# Set this to the folder containing your pictures.
#
# Examples:
# (Windows) folder_path = r"C:\Users\YourName\Desktop\MyPhotos"
# (Mac/Linux) folder_path = "/Users/YourName/Desktop/MyPhotos"
# -----------------------------------------------------------------
folder_path = r"I:\UCA\intro to deep learning\BearCar\data\2025-11-12-14-00\images"


# --- No need to edit below this line ---

# Define a list of common image file extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']

# --- WARNING ---
# This script OVERWRITES your original files.
# Make a backup first if you are unsure.
# --- WARNING ---

try:
    # Convert the string path to a Path object
    target_folder = Path(folder_path)

    if not target_folder.is_dir():
        print(f"Error: Path '{folder_path}' is not a valid directory.")
        exit()

    print(f"Scanning folder: {target_folder}")
    print("WARNING: Files will be permanently overwritten.\n")

    # Loop through all files in the target folder
    found_images = 0
    for file_path in target_folder.iterdir():
        # Check if it's a file and has a valid image extension
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                # Open the image using a 'with' block
                with Image.open(file_path) as img:
                    # Rotate the image 180 degrees
                    rotated_img = img.rotate(180)
                                        
                    # Save the rotated image OVERWRITING the original
                    rotated_img.save(file_path)
                    print(f"✅ Rotated (overwritten): {file_path.name}")
                    found_images += 1

            except Exception as e:
                print(f"❌ Could not rotate {file_path.name}: {e}")
        
        elif file_path.is_file():
            print(f"⚪ Skipping (not an image): {file_path.name}")

    print(f"\n--- Done! ---")
    if found_images == 0:
        print("No image files were found in the specified folder.")
    else:
        print(f"Successfully rotated and overwrote {found_images} image(s).")

except Exception as e:
    print(f"An unexpected error occurred: {e}")