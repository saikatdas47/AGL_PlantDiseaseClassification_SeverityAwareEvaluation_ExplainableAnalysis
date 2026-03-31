import os
import random
import shutil

# ======================
# YOUR PATHS
# ======================
SOURCE_DIR = "/Users/saikatdas/Desktop/CV/Data"
OUTPUT_DIR = "/Users/saikatdas/Desktop/CV/Datasplit"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)

# ======================
# CREATE FOLDERS
# ======================
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for split in ["train", "val", "test"]:
    for cls in classes:
        path = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(path, exist_ok=True)

# ======================
# SPLIT LOGIC
# ======================
for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)

    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end   = train_end + int(total * VAL_RATIO)

    train_files = images[:train_end]
    val_files   = images[train_end:val_end]
    test_files  = images[val_end:]

    def copy_files(file_list, split):
        for f in file_list:
            src = os.path.join(class_path, f)
            dst = os.path.join(OUTPUT_DIR, split, cls, f)
            shutil.copy2(src, dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("✅ Dataset split completed!")