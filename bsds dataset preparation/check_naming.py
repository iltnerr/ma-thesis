"""Check naming of cleaned edge files in case something went wrong while saving .png files"""

from matplotlib import pyplot as plt
import os


def check_name(name, dir):
    for root, dirs, files in os.walk("../data/BSDS500_groundTruth/" + dir):
        if not name in files:
            print("couldn't find file named '" + name + "' in directory '" + dir + "'")


for directory in ["test", "train", "val"]:
    for root, dirs, files in os.walk("../data/clear edges/" + directory):
        for filename in files:
            check_name(filename, directory)

print("finished checking.")
