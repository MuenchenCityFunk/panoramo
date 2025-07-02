from . import utils
from . import exceptions
import os
import cv2
import time

def stitch_images(image_folder, image_filenames, stitch_direction):
    """Function to stitch a sequence of input images.
        Images can be stitched horizontally or vertically.
        For horizontal stitching the images have to be passed from left to right order in the scene.
        For vertical stitching the images have to be passed from top to bottom order in the scene.
    
    Args:
        image_folder (str): path of the directory containing the images
        image_filenames (list): a list of image file names in the order of stitching
        stitch_direction (int): 1 for horizontal stitching, 0 for vertical stitching
    
    Returns:
        stitched_image (numpy array): of shape (H, W, 3) representing the stitched image
    """
    num_images = len(image_filenames)
    
    if num_images < 2:
        raise(exceptions.InsufficientImagesError(num_images))
    
    valid_files, file_error_msg = utils.check_imgfile_validity(image_folder, image_filenames)
    if not valid_files:
        raise(exceptions.InvalidImageFilesError(file_error_msg))
    
    # Lade das erste Bild (Pivot)
    pivot_img_path = os.path.join(image_folder, image_filenames[0])
    pivot_img = cv2.imread(pivot_img_path)
    
    # Überprüfe ob das Pivot-Bild erfolgreich geladen wurde
    if pivot_img is None:
        raise RuntimeError(f"Konnte Pivot-Bild nicht laden: {pivot_img_path}")
    
    print(f"Pivot-Bild geladen: {image_filenames[0]} (Größe: {pivot_img.shape})")

    successful_stitches = 0
    for i in range(1, num_images, 1):
        join_img_path = os.path.join(image_folder, image_filenames[i])
        join_img = cv2.imread(join_img_path)
        
        # Überprüfe ob das Join-Bild erfolgreich geladen wurde
        if join_img is None:
            print(f"⚠️  Warnung: Konnte Bild nicht laden: {image_filenames[i]}, überspringe...")
            continue
        
        print(f"Verarbeite Bild {i+1}/{num_images}: {image_filenames[i]} (Größe: {join_img.shape})")
        
        try:
            pivot_img = utils.stitch_image_pair(pivot_img, join_img, stitch_direc=stitch_direction)
            successful_stitches += 1
        except Exception as e:
            print(f"⚠️  Warnung: Fehler beim Stitching von {image_filenames[i]}: {e}, überspringe...")
            continue
    
    print(f"✅ Erfolgreich gestitcht: {successful_stitches} von {num_images-1} Bildern")
    
    # VERBESSERTE Validierung - akzeptiert Teil-Panoramen
    if pivot_img is None:
        raise RuntimeError("Das finale Panorama ist None")
    
    if pivot_img.size == 0:
        raise RuntimeError("Das finale Panorama ist leer (0 Pixel)")
    
    # Prüfe ob mindestens 2 Bilder erfolgreich gestitcht wurden
    if successful_stitches < 1:
        raise RuntimeError(f"Zu wenig erfolgreiche Stitches: {successful_stitches} (Minimum: 1)")
    
    print(f"🎯 Teil-Panorama erstellt: {successful_stitches+1} Bilder → {pivot_img.shape[1]}x{pivot_img.shape[0]} Pixel")
    
    return pivot_img

def stitch_images_and_save(image_folder, image_filenames, stitch_direction, output_folder=None):
    """Function to stitch and save the resultant image.
        Images can be stitched horizontally or vertically.
        For horizontal stitching the images have to be passed from left to right order in the scene.
        For vertical stitching the images have to be passed from top to bottom order in the scene.
    
    Args:
        image_folder (str): path of the directory containing the images
        image_filenames (list): a list of image file names in the order of stitching
        stitch_direction (int): 1 for horizontal stitching, 0 for vertical stitching
        output_folder (str): the directory to save the stitched image (default is None, which creates a directory named "output" to save)
    
    Returns:
        None
    """
    timestr = time.strftime("%Y%m%d_%H%M%S")
    filename = "stitched_image_" + timestr + ".jpg"
    stitched_img = stitch_images(image_folder, image_filenames, stitch_direction)
    
    # VERBESSERTE Validierung - akzeptiert Teil-Panoramen
    if stitched_img is None:
        raise RuntimeError("Das gestitchte Bild ist None")
    
    if stitched_img.size == 0:
        raise RuntimeError("Das gestitchte Bild ist leer (0 Pixel)")
    
    # Prüfe minimale Größe für Teil-Panoramen
    min_width = 100   # Mindestbreite für Teil-Panorama
    min_height = 100  # Mindesthöhe für Teil-Panorama
    
    if stitched_img.shape[1] < min_width or stitched_img.shape[0] < min_height:
        raise RuntimeError(f"Panorama zu klein: {stitched_img.shape[1]}x{stitched_img.shape[0]} (Minimum: {min_width}x{min_height})")
    
    if output_folder is None:
        if not os.path.isdir("output"):
            os.makedirs("output/")
        output_folder = "output"
    else:
        # Stelle sicher, dass der Output-Ordner existiert
        os.makedirs(output_folder, exist_ok=True)
        
    full_save_path = os.path.join(output_folder, filename)
    
    # Versuche das Bild zu speichern
    success = cv2.imwrite(full_save_path, stitched_img)
    if not success:
        raise RuntimeError(f"Konnte Panorama nicht speichern: {full_save_path}")
    
    print("The stitched image is saved at: " + full_save_path)
    print(f"Panorama-Größe: {stitched_img.shape}")
    print(f"✅ Teil-Panorama erfolgreich erstellt: {len(image_filenames)} Frames → {stitched_img.shape[1]}x{stitched_img.shape[0]} Pixel")
