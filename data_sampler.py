import os
import random
import shutil

SHOULD_FILES_BE_MOVED = False
SOURCE_DIRECTORY = "/home/stefan/Uni/Master/Semester_3/seminar_project/CelebA/Img/img_align_celeba_minus_50K_images"
DESTINATION_DIRECTORY = "/home/stefan/Uni/Master/Semester_3/seminar_project/CelebA/Img/img_align_celeba_50K_samples"
files_list = []

for root, dirs, files in os.walk(SOURCE_DIRECTORY):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            files_list.append(os.path.join(root, file))
files_to_move = random.sample(files_list, 50_000)

# print(files_to_move)
# print(len(files_list))
# print(files_list[:5])

if SHOULD_FILES_BE_MOVED:
    if not os.path.isdir(DESTINATION_DIRECTORY):
        os.makedirs(DESTINATION_DIRECTORY)

    for file in files_to_move:
        shutil.move(file, DESTINATION_DIRECTORY)
