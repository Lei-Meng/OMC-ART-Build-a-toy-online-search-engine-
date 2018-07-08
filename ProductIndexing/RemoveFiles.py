import os
import shutil


def RemoveFiles(folder_path):

    feature_files = os.listdir(folder_path)
    for files in feature_files:
        try:
              os.unlink(folder_path + files)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
