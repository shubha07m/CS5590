import os
import random
import shutil

# Set the root directory where the classes folders are located

datadir_name = 'Archive'
base_dir = os.path.join(os.getcwd(), datadir_name)

# Set the percentages for train, test, and validation data
train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2


def data_divider(div_dict):
    if '.DS_Store' in os.listdir(base_dir):
        os.remove(os.path.join(base_dir, '.DS_Store'))

    # Set the names of the folders
    classes = os.listdir(base_dir)

    # Loop through each folder
    for part in div_dict.keys():

        # Set the path to the current folder

        folder_path = os.path.join(base_dir, part)

        for class_id in classes:
            class_path = os.path.join(base_dir, class_id)
            if '.DS_Store' in os.listdir(class_path):
                os.remove(os.path.join(class_path, '.DS_Store'))

            # Create the directory inside the current folder
            part_path = os.path.join(folder_path, class_id)
            os.makedirs(part_path, exist_ok=True)

            # Get the list of image files in the current folder
            image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            # Shuffle the list of image files
            random.shuffle(image_files)

            # Calculate the number of images for each set based on the ratios
            num_images = len(image_files)
            # Assign images to the train, test, and validation sets

            class_files = image_files[:int(num_images * div_dict[part])]
            #
            # # Copy images to the respective directories
            for file in class_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(part_path, file)
                shutil.copy(src, dst)

            print("Images of class %s for part %s copied successfully!" % (class_id, part))


if __name__ == "__main__":
    divdict = {'train': train_ratio, 'test': test_ratio, 'valid': valid_ratio}
    data_divider(divdict)
