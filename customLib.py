import os
from PIL import Image
#from PIL import Image, ImageDraw
#import pandas as pd

def crop_image(image, xy=(50, 50), wh=(640, 640)):
    # Crop (x,y,width,height) from PIL image
    x, y = xy
    right, bottom = tuple(a + b for a, b in zip(wh, xy))
    image = image.crop((x, y, right, bottom))
    return image

def get_labelIn_window(
    section_xy, section_wh, orig_label_xy, orig_label_wh, label_number=0
):
    # Unpack inputs
    section_x, section_y = section_xy
    section_w, section_h = section_wh
    orig_label_x, orig_label_y = orig_label_xy
    orig_label_w, orig_label_h = orig_label_wh

    # Convert original bbox center->top-left relative to window origin
    new_label_x = (orig_label_x - (orig_label_w / 2)) - section_x
    new_label_y = (orig_label_y - (orig_label_h / 2)) - section_y

    # Normalize to window size (YOLO format components)
    norm_new_label_x = round(new_label_x / section_w, 6)
    norm_new_label_y = round(new_label_y / section_h, 6)
    norm_orig_label_w = round(orig_label_w / section_w, 6)
    norm_orig_label_h = round(orig_label_h / section_h, 6)
    label_coordonates = f"{label_number} {norm_new_label_x} {norm_new_label_y} {norm_orig_label_w} {norm_orig_label_h}"
    
    # Keep only boxes fully inside window (non-negative, within bounds)
    is_visible_label = (
        ((orig_label_w / 2) + new_label_x)
        <= section_w  # 70% of the label is inside the width box
        and ((orig_label_h / 2) + new_label_y)
        <= section_h  # 70% of the label is inside the height box
        and norm_new_label_x >= 0
        and norm_new_label_y >= 0
        and norm_orig_label_w >= 0
        and norm_orig_label_h >= 0
    )
    # return label coordonates if the label is fully visible and empty string if not
    return label_coordonates if is_visible_label else ""

def generate_windows(input_folder, dest_path, window_wh=(640, 640), stride_percent=0.5):
     # Sliding-window tiling with overlap; save chips + YOLO labels
    files = os.listdir(input_folder + "/images")
    window_width, window_height = window_wh
    stride = int(window_width * stride_percent)
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    count = 0
    # cycle trough images
    for image_name in image_files:
        # image object
        image = Image.open(f"{input_folder}/images/{image_name}")
        [img_name, img_extension] = image_name.rsplit(".", 1)
        image_width, image_height = image.size

        # Number of steps in each axis (rounded down)
        w_windows = int(image_width / stride)  # nr of strides in image width
        h_windows = int(image_height / stride)  # nr of strides in image height
        count = 0  # incrementing to name the files uniquely
        
        for y in range(h_windows):
            is_last_y_window = False
            # detecting the last y iteration
            if y + 1 == h_windows:
                # setting a variable to know it is the last iteration
                is_last_y_window = True
                # y position window logic
            new_slice_y = (
                image_height - window_height  # last window to the right
                if is_last_y_window == True  # if it's the last itaration
                else y * stride  # normal stride calculation for y
            )
            
            # cycling trough sections in height
            for x in range(w_windows):
                # # Snap last col to right edge
                # if this is the last horizontal window w_section iteration
                is_last_x_window = False
                if x + 1 == w_windows:
                    is_last_x_window = True
                    
                new_slice_x = (
                    int(image_width - window_width)
                    if is_last_x_window == True
                    else x * stride
                )
                
                # crop chip
                new_slice_xy = (new_slice_x, new_slice_y)
                new_slice_wh = (window_width, window_height)
                section_image = crop_image(
                    image,
                    xy=new_slice_xy,
                    wh=new_slice_wh,
                )

                # save image file in output folder with _count appended postfix name
                image_dest_file = (
                    f"{dest_path}images/{img_name}_{count}.{img_extension}"
                )
                # save label destination path
                label_dest_file = f"{dest_path}/labels/{img_name}_{count}.txt"
                
                # Save chip
                section_image.save(image_dest_file)
                
                # Read original labels
                with open(f"{input_folder}/labels/{img_name}.txt", "r") as file:
                    # Read all lines into a list
                    lines = file.readlines()

                # Strip newline characters from each line (optional)
                labels = [line.strip() for line in lines]
                # cycle trough labels
                index = 0
                final_label_text = ""
                for label_data_string in labels:
                    label_data_list = label_data_string.split(" ")
                    label_wh = (
                        (float(label_data_list[3]) * image_width),
                        (float(label_data_list[4]) * image_height),
                    )
                    label_w, label_h = label_wh
                    # Convert normalized center to absolute center, then to our helper inputs
                    label_xy = (
                        (float(label_data_list[1]) * image_width) + (label_w / 2),
                        (float(label_data_list[2]) * image_height) + (label_h / 2),
                    )

                    # get the label that's possibly inside this window
                    label_in_window = get_labelIn_window(
                        section_xy=(new_slice_x, new_slice_y),
                        section_wh=window_wh,
                        orig_label_xy=label_xy,
                        orig_label_wh=label_wh,
                        label_number=index,
                    )

                    if len(label_in_window) > 0:
                        final_label_text += label_in_window + "\n"
                    else:
                        final_label_text = label_in_window

                # save the label for the image:
                with open(label_dest_file, "w") as file:
                    file.write(final_label_text)
                count += 1