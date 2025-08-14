from customLib import *

# Generate tiled windows and YOLO labels
def main(input_folder, dest_path):
    generate_windows(input_folder=input_folder, dest_path=dest_path)

source_path = "./labelled_data/"
#dest_path = "./datasets/train/" # switch to train split when needed
dest_path = "./datasets/val/"    # switch to val split when needed

# Build tiled dataset; pass test_file_name in generate_windows to debug a single page
main(input_folder=source_path, dest_path=dest_path)  # , test_file_name="14_1696"