import os
import shutil
import re
from collections import defaultdict

# CONFIGURATION
original_root = "/home/student/edgemapper-client/train_data/nyu_data/data/nyu2_train"           # Replace with your actual root
copy_root = original_root + "_split"            # New copied directory
files_per_group = 60                          # Number of images per subdirectory

# Step 1: Copy the entire root directory to a new one
if os.path.exists(copy_root):
    raise FileExistsError(f"{copy_root} already exists. Delete it or choose another name.")
shutil.copytree(original_root, copy_root)
print(f"Copied {original_root} → {copy_root}")

# Step 2: Function to get all image files (sorted)
def get_image_files(folder):
    def extract_number(filename):
        match = re.match(r"(\d+)", os.path.splitext(filename)[0])
        return int(match.group(1)) if match else float('inf')

    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and os.path.isfile(os.path.join(folder, f))
    ], key=extract_number)

# Step 3: Walk through subdirectories of the copy
for subdir, dirs, files in os.walk(copy_root):
    if subdir == copy_root:
        continue  # skip the root itself

    image_files = get_image_files(subdir)
    if not image_files:
        continue

    num_groups = (len(image_files) + files_per_group - 1) // files_per_group


    # Helper to group by numeric prefix
    def group_images_by_prefix(files):
        grouped = defaultdict(dict)
        for f in files:
            name, ext = os.path.splitext(f)
            if name.isdigit():
                grouped[int(name)][ext.lower()] = f
        return dict(sorted(grouped.items()))

    for i in range(num_groups):
        group_dir = os.path.join(subdir, f"part_{i+1:03d}")
        os.makedirs(group_dir, exist_ok=True)

        start = i * files_per_group
        end = min(start + files_per_group, len(image_files))

        image_subset = image_files[start:end]
        grouped_images = group_images_by_prefix(image_subset)

        for idx, (prefix, file_dict) in enumerate(grouped_images.items(), start=1):
            for ext, original_file in file_dict.items():
                new_name = f"{idx}{ext}"
                src = os.path.join(subdir, original_file)
                dst = os.path.join(group_dir, new_name)
                shutil.move(src, dst)



    print(f"Processed {subdir} → {num_groups} parts")

print("✅ Done. Original preserved, split version in:", copy_root)
