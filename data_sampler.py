import os
import random
import shutil

SAMPLE_COUNT = 256 - 64
SHOULD_FILES_BE_MOVED = True
SHOULD_FILES_BE_COPIED = False
SOURCE_DIRECTORY = "/home/stefan/Uni/Master/Semester_3/seminar_project/CelebA/Img/256_images/img_align_celeba_minus_256_images"
DESTINATION_DIRECTORY = "/home/stefan/Uni/Master/Semester_3/seminar_project/CelebA/Img/256_images/img_align_celeba_256_samples"
files_list = []

for root, dirs, files in os.walk(SOURCE_DIRECTORY):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            files_list.append(os.path.join(root, file))
files_to_move = random.sample(files_list, SAMPLE_COUNT)


if SHOULD_FILES_BE_MOVED:
    if not os.path.isdir(DESTINATION_DIRECTORY):
        os.makedirs(DESTINATION_DIRECTORY)

    for file in files_to_move:
        shutil.move(file, DESTINATION_DIRECTORY) 
        
       
if SHOULD_FILES_BE_COPIED:
    if not os.path.isdir(DESTINATION_DIRECTORY):
        os.makedirs(DESTINATION_DIRECTORY)

    for file in files_to_move:
        shutil.copy(file, DESTINATION_DIRECTORY) 
