import os
from PIL import Image
import random

# Use https://online.photoscissors.com/ to cut out head
# generate imgs with data to get more training data
def generating_waldos(use_background=True):
    background_use_num = 2
    head_use_num = 2
    size = 640
    im_num = 0
    # Getting head
    head_dir = "waldoData/Clean/OnlyWaldoHeads"
    back_dir = "waldoData/Clean/ClearedWaldos"
    
    # Get all Waldo head images
    head_images = [f for f in os.listdir(head_dir) if f.lower().endswith(('.png', '.jpg'))]
    
     # Get all background images
    back_images = [f for f in os.listdir(back_dir) if f.lower().endswith(('.png', '.jpg'))]

    
    for head_name in head_images:
        for _ in range(head_use_num):
                for back_name in back_images:
                    for _ in range(background_use_num):
                        # Load head
                        foreground = Image.open(os.path.join(head_dir, head_name)).convert("RGBA")
                        # Optional rotation
                        if random.randint(0, 9) < 5:
                            num = random.randint(-15, 15)
                            foreground = foreground.rotate(num, expand=True)
                        # Random scaling
                        if random.randint(0, 9) < 7:
                            scale = random.uniform(0.8, 1.5)
                            w, h = foreground.size
                            foreground = foreground.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                    # Load background
                    background = Image.open(os.path.join(back_dir, back_name)).convert("RGBA")

                    bck_w, bck_h = background.size

                    # Skip small backgrounds
                    if bck_w < size or bck_h < size:
                        print(f"Skipping background {back_name}: too small ({bck_w}x{bck_h})")
                        continue

                    # Pick random crop from background
                    bck_x = random.randint(0, bck_w - size)
                    bck_y = random.randint(0, bck_h - size)
                    cropped = background.crop((bck_x, bck_y, bck_x + size, bck_y + size))

                    # Save background-only crop (negative sample)
                    cropped.convert("RGB").save(f"Data/NotWaldo_1/n{im_num}.png")

                    # Place foreground at random position inside crop
                    frg_w, frg_h = foreground.size
                    if frg_w < size and frg_h < size:
                        frg_x = random.randint(0, size - frg_w)
                        frg_y = random.randint(0, size - frg_h)
                        cropped.paste(foreground, (frg_x, frg_y), foreground)

                    # Save positive sample
                    cropped.convert("RGB").save(f"Data/Waldo_1/{im_num}{use_background}.png")

                    im_num += 1

    print(f"Generated {im_num} images.")

generating_waldos(use_background=True)