
from onlinesearchengine import onlinesearchengine
from VFExtractor import VFExtractor
import shutil

if __name__ == "__main__":

    root = 'C:\\Users\\lily\\Desktop\\'

    database_path = root + 'database\\'
    img_dir = database_path + 'images\\'

    direc = img_dir + '12.jpg'
    text = ['protein']
    k = 10 #number of search results

    root_files = root + 'indexing_data\\'
    index, simlist = onlinesearchengine(direc,text,k,root_files)
    print(index,simlist)

    for i in range(0,k):
        shutil.copy2(img_dir + str(int(index[i])+1)+ '.jpg', root + 'results')
