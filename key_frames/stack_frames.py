import argparse
import cv2
from pathlib import Path
import numpy as np
import os

def main(base_dir):
    """
    Stack min and max mean keyframes together to form a single represention."
    Args;
        base_dir - Base directory tree containing keyframes.
    """


    indir = Path(base_dir)

    subjects = list(indir.glob("*"))

    lowest_dirs = []

    #get folders containing keyframes
    for root,dirs,files in os.walk(indir):
        if files and not dirs:
            lowest_dirs.append(root)
    
    lowest_dirs.sort()
    print(lowest_dirs)
    del lowest_dirs[-1]

    for d in lowest_dirs:
        print(d)

        try:
            min_mean_frame = cv2.imread(d + "/min_mean_keyframe.png",0)
            max_mean_frame = cv2.imread(d + "/max_mean_keyframe.png",0)
            stack_frame = np.hstack([min_mean_frame,max_mean_frame])
            cv2.imwrite(d + "/stack_mean_keyframe.png",stack_frame)
        except:
            print ("No frames")


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Stack min and max mean frames to form a single image.")
    
    parser.add_argument(
            'base_dir',
            help="Base directory tree containing silhouette keyframes.")

    args = parser.parse_args()

    main(args.base_dir)
