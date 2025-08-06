import os
import shutil
import random
from tqdm import tqdm

# === CONFIG ===
DOCS_DIR = 'DocXPand-25k/documents'
OUTPUT_DIR = 'project_data'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SAMPLE_LIMIT = 2500  # 💡 Limit to 2.5k per class

# === Define real and fake folders based on paper ===
REAL_FOLDERS = {
    "ID_CARD_TD1_A", "ID_CARD_TD2_A",
    "PP_TD3_A", "PP_TD3_C",
    "RP_CARD_TD1", "RP_CARD_TD2"
}

FAKE_FOLDERS = {
    "ID_CARD_TD1_B", "ID_CARD_TD2_B",
    "PP_TD3_B"
}

# === Setup output folders ===
splits = ['train', 'val', 'test']
labels = ['real', 'fake']

for split in splits:
    for label in labels:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# === Collect all image paths and labels ===
all_samples = {'real': [], 'fake': []}

for folder in os.listdir(DOCS_DIR):
    folder_path = os.path.join(DOCS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    label = None
    if folder in REAL_FOLDERS:
        label = 'real'
    elif folder in FAKE_FOLDERS:
        label = 'fake'
    else:
        print(f"⚠️  Skipping unknown folder: {folder}")
        continue

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, fname)
            all_samples[label].append(full_path)

# === Shuffle and split ===
for label in labels:
    samples = all_samples[label]
    random.shuffle(samples)

    # 💡 Limit sample size to avoid overload
    samples = samples[:SAMPLE_LIMIT]

    total = len(samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_map = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }

    # === Copy files ===
    for split in splits:
        for src_path in tqdm(split_map[split], desc=f'{label.upper()} → {split}'):
            dst_path = os.path.join(OUTPUT_DIR, split, label, os.path.basename(src_path))
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"[⚠️] Failed to copy {src_path}: {e}")
