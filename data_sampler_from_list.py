import os
import random
import shutil

SHOULD_FILES_BE_MOVED = False
SHOULD_FILES_BE_COPIED = True
FILENAME = os.path.join("/home/stefan/Uni/Master/Semester_3/seminar_project/GAN-Leaks-master/gan_model_checkpoints/vaegan_identity0_20k/", "subset_identity_0_20k.txt")
SOURCE_DIRECTORY = "/home/stefan/Uni/Master/Semester_3/seminar_project/CelebA/Img/img_align_celeba"
DESTINATION_DIRECTORY = "/home/stefan/Uni/Master/Semester_3/seminar_project/CelebA/Img/img_align_celeba_20K_samples_for_vaegan"
files_list = []

with open(FILENAME) as file:
    for line in file:
        files_list.append(os.path.join(SOURCE_DIRECTORY, line.rstrip().split(" ", 1)[1]))
files_to_move = files_list

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
