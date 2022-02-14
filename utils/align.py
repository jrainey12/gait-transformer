import numpy as np
import cv2
from glob import glob
import argparse
from os.path import join, basename, exists
import os
import cython
#from get_bounds import get_bounds
from bounds.get_bounds import bounds

def main(base_dir):
    """
    Iterate through the silhouette frames align the silhouette with the centre of the frame.
    Args:
        base_dir (str): directory tree containing silhouette data.
    """

    lowest_dirs = []

    #get lowest dirs which contain sils
    for root,dirs,files in os.walk(base_dir):
        if files and not dirs:
            lowest_dirs.append(root)

    lowest_dirs.sort()

    for sil_dir in lowest_dirs:
        
        if basename(sil_dir) == "aligned":
            print ("Already Aligned")
            continue

        print ("In Dir: ", sil_dir)
        img_paths = sorted(glob(join(sil_dir, "*.png")))
        if not img_paths == []:
            for path in img_paths:
                align(sil_dir, path, pad_image(path))


def align(sil_dir, img_path, p_img):
    """
    Align the silhouettes.
    Args:
        sil_dir (str): directory containing sils.
        img_path (str): path to image.
        p_img (ndarray): padded image.
    Returns:
        aligned_img (ndarray): final aligned image.
    """

    #@cython.boundscheck(False)
    top,bottom = bounds(p_img)

    #print top,bottom
    h,w = p_img.shape
    #calc height of sil
    sil_height = bottom[1] - top[1]

    #centre of gravity of silhouette
    cnts = cv2.findContours(p_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            print ("incomplete contour")
            continue

        else:
            X = int(M["m10"] / M["m00"])
            Y = int(M["m01"] / M["m00"])

    middle = (X,Y)

    #Get bounding box parameters
    box_width = (11 * sil_height) / 16

    box_right = int(middle[0] + (box_width/2))

    if box_right > w:
        print (box_right)
        print ("ERROR")
        return 0,0,0,0
    
    box_left = int(middle[0] - (box_width/2))

    if box_left < 0:
        print (box_left)
        print ("ERROR")
        return 0,0,0,0

    box_top = int(top[1])
    box_bottom = int(bottom[1])


    #crop image to bounding box
    crop_img = p_img[box_bottom:box_top, box_right:box_left] 
    #print(crop_img.shape)
    
    #resize image to standard size
    try:
        out_img = cv2.resize(crop_img, (88,128))
    except:
        print ("Failed to crop.")
        return

    #create output folder
    outpath = join(sil_dir, "aligned")
    if not exists(outpath):
        os.mkdir(outpath)

    #Save image
    out_dir = join(outpath, basename(img_path))
    #print ("Out Dir: ",out_dir)
    cv2.imwrite(out_dir, out_img)

def pad_image(img_path):
    """
    Pad an image at the left and right to allow space for correct cropping.
    Args:
        img_path (str): path to image

    Return:
        padded_img (ndarray): image with 50 pixel padding.
    """

    #read in image
    img = cv2.imread(img_path, 0)

    padded_img = cv2.copyMakeBorder(img,0,0,50,50,cv2.BORDER_CONSTANT,value=[000])

    return padded_img


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Align silhouettes to centre of frame and crop.")

    parser.add_argument(
            'base_dir',
            help="Directory tree of silhouette data.")
    
    args = parser.parse_args()
    main(args.base_dir)



