import os
import shutil
import random
import pandas as pd

source_images = 'images_256_256' 
base_dir = 'dataset'

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

for split in['train', 'validation', 'test']:
    for category in['Whites', 'Darks', 'Colors']:
        os.makedirs(os.path.join(base_dir, split, category))

print("Folders created successfully.")


print(f"Scanning '{source_images}' for pictures...")
image_phonebook = {} 

for root, dirs, files in os.walk(source_images):
    for file_name in files:
        if file_name.endswith('.jpg'):
            full_path = os.path.join(root, file_name)
            image_phonebook[file_name] = full_path

if len(image_phonebook) == 0:
    print("Error: 0 images found. Make sure the folder is named correctly!")
    exit()

print("Reading H&M's CSV file...")
df = pd.read_csv('articles.csv', dtype={'article_id': str})

whites_list = []
darks_list = []
colors_list =[]


print("Applying sorting rules...")

for index, row in df.iterrows():
    item_type = str(row['product_group_name'])
    if item_type not in['Garment Upper body', 'Garment Lower body']:
        continue 
        
    color = str(row['colour_group_name'])
    shade = str(row['perceived_colour_value_name']) 
    pattern = str(row['graphical_appearance_name']) 


    if pattern not in ['Solid', 'Placement print']:
        continue


    if color == 'White':
        category = 'Whites'
        

    elif shade == 'Dark' or color in['Black', 'Navy Blue', 'Dark Grey', 'Charcoal']:
        category = 'Darks'
        

    elif color in['Red', 'Pink', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange', 'Magenta', 'Light Blue']:
        category = 'Colors'
        
    else:
        continue 


    article_id = str(row['article_id']).zfill(10)
    img_name = article_id + ".jpg"
    
    if img_name in image_phonebook:
        img_path = image_phonebook[img_name]
        
        if category == 'Whites':
            whites_list.append(img_path)
        elif category == 'Darks':
            darks_list.append(img_path)
        elif category == 'Colors':
            colors_list.append(img_path)


print(" PERFECT DATASET COUNTS")
print(f" WHITES: {len(whites_list)} perfect images")
print(f" DARKS:  {len(darks_list)} perfect images")
print(f" COLORS: {len(colors_list)} perfect images")

min_count = min(len(whites_list), len(darks_list), len(colors_list))

if min_count == 0:
    print("\nError: The rules deleted everything! Check your dataset.")
    exit()

print(f"\nBalancing the dataset... We will use exactly {min_count} images per class.")

random.seed(42)
whites_balanced = random.sample(whites_list, min_count)
darks_balanced = random.sample(darks_list, min_count)
colors_balanced = random.sample(colors_list, min_count)

num_train = int(min_count * 0.70)
num_val = int(min_count * 0.15)
num_test = min_count - num_train - num_val


print("Copying files into Train, Validation, and Test folders...")

def copy_images(image_list, category_name):
    for i in range(0, num_train):
        shutil.copy(image_list[i], os.path.join(base_dir, 'train', category_name, os.path.basename(image_list[i])))
        
    for i in range(num_train, num_train + num_val):
        shutil.copy(image_list[i], os.path.join(base_dir, 'validation', category_name, os.path.basename(image_list[i])))
        
    for i in range(num_train + num_val, min_count):
        shutil.copy(image_list[i], os.path.join(base_dir, 'test', category_name, os.path.basename(image_list[i])))

    print(f"[{category_name}] files successfully copied!")

copy_images(whites_balanced, 'Whites')
copy_images(darks_balanced, 'Darks')
copy_images(colors_balanced, 'Colors')

print("\nDone. Dataset is prepared.")