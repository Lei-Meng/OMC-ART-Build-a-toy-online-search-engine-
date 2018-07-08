import os
import shutil


def Backup(indexing_path,backup_path):

    indexing_files = os.listdir(indexing_path)
    for files in indexing_files:
        shutil.copy(indexing_path + files, backup_path)

    return 0