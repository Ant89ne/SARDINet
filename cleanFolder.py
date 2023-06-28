import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('folderToClean', type=str, help='Folder to clean')
args = parser.parse_args()

def clean(folder):
    files = [folder + '/' + f for f in os.listdir(folder) if f not in ['.', '..']]
    patterns = ['_100.', '_300.', '_500.', '_1000.', '_2000.','input']

    for f in files : 
        flag = False
        if os.path.isdir(f):
            clean(f)

        elif os.path.isfile(f):
            name = f[f.rfind('/'):]

            for p in patterns :
                if p in name : 
                    flag = True
                    break
            if not flag : 
                os.remove(f)


clean(args.folderToClean)