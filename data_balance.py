import os
import random
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.
    
    Args:
        filename (str): The name of the file to check.
        
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions

def balance_datasets(base_path, target_count):
    """
    Balances the number of images in each subfolder of a given directory to a target count.
    Only processes image files and preserves the original file structure.

    Args:
        base_path (str): The path to the directory containing the dataset folders.
        target_count (int): The desired number of images in each folder (melanose folder count: 2600).
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return
    
    logger.info(f"Starting dataset balancing with target count: {target_count}")
    logger.info(f"Processing directory: {base_path}")
    
    total_removed = 0
    processed_folders = 0
    
    # Process each subfolder
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            folder_name = folder_path.name
            logger.info(f"Processing folder: {folder_name}")
            
            # Get all image files in the folder
            image_files = [f for f in folder_path.iterdir() 
                          if f.is_file() and is_image_file(f.name)]
            
            current_count = len(image_files)
            logger.info(f"Found {current_count} images in {folder_name}")
            
            if current_count > target_count:
                # Calculate how many images to remove
                excess_count = current_count - target_count
                
                # Randomly select images to remove (unbiased selection)
                random.seed()  # Ensure truly random selection
                images_to_remove = random.sample(image_files, excess_count)
                
                # Remove the selected images
                removed_count = 0
                for image_file in images_to_remove:
                    try:
                        image_file.unlink()  # Remove the file
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {image_file}: {e}")
                
                total_removed += removed_count
                logger.info(f"Removed {removed_count} images from {folder_name} (target: {excess_count})")
                
            elif current_count == target_count:
                logger.info(f"Folder {folder_name} already has the target count ({target_count})")
            else:
                logger.info(f"Folder {folder_name} has fewer images ({current_count}) than target ({target_count})")
            
            processed_folders += 1
    
    logger.info(f"Dataset balancing completed!")
    logger.info(f"Processed {processed_folders} folders")
    logger.info(f"Total images removed: {total_removed}")

def verify_balance(base_path, target_count):
    """
    Verify the balance of images across folders after balancing.
    
    Args:
        base_path (str): The path to the directory containing the dataset folders.
        target_count (int): The expected number of images in each folder.
    """
    base_path = Path(base_path)
    logger.info("\n=== VERIFICATION REPORT ===")
    
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            image_files = [f for f in folder_path.iterdir() 
                          if f.is_file() and is_image_file(f.name)]
            current_count = len(image_files)
            status = "✓ BALANCED" if current_count <= target_count else "⚠ EXCESS"
            logger.info(f"{folder_path.name}: {current_count} images {status}")

if __name__ == '__main__':
    # Path to the directory containing the dataset folders
    dataset_path = r"C:\Users\shynn\Applications\datasets\orange_project_datasets\raw_data"
    
    # The number of images in the 'melanose' folder (target count)
    melanose_count = 2600
    
    # Balance the datasets
    balance_datasets(dataset_path, melanose_count)
    
    # Verify the results
    verify_balance(dataset_path, melanose_count)