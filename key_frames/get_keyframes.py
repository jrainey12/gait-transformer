import argparse
import os
from pathlib import Path
import cv2
from shutil import copyfile
import numpy as np

def main(base_dir):

    """
    Find the keyframes from a sequence of aligned gait silhouettes.
    A keyframe occurs at minimum and maximum stride lengths, in the middle of each phase of a gait cycle.

    Args:
        base_dir - Directory tree containing aligned gait silhouette sequences in the lowest directories.
        out_dir - Directory to save final keyframes (input dir tree layout will be recreated).
    """

    indir = Path(base_dir)

    subjects = list(indir.glob("*"))
   
    lowest_dirs = []
    
    #get all folders containing sil sequences 
    for root,dirs,files in os.walk(indir):
        if files and not dirs:
            lowest_dirs.append(root)
    
    lowest_dirs.sort()
    
    #delete last record, zips folder.
    del lowest_dirs[-1]
    for d in lowest_dirs:
        print("Directory: ", d)
        try:
            find_keyframes(d) 
        except:
            print("No frames!")

def find_keyframes(sil_dir):
    """
    Find the keyframes using the stride lengths in each frame.
    
    Args:
        sil_dir - Directory containing silhouettes.

    Return:
        keyframes - list of file paths of key frames.
    """
    
    sil_dir = Path(sil_dir)
    
    sils = sorted(list(sil_dir.glob("*")))
    
    #get all stride widths
    stride_widths = []

    for sil in sils:
        try:
            stride_widths.append(get_stride_width(sil))
        except:
            continue
    #print(stride_widths)
   #find min and max strides 
    minmax = find_minima(stride_widths,0)
    #maxima = find_maxima(stride_widths)

    #print(minmax)

    keyframes = sorted([str(sils[i]) for i in minmax])

    #print(keyframes)
    out_dir = os.path.dirname(keyframes[0]) + "/keyframes/" 
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, kf in enumerate(keyframes):
        #print(kf)
        outname = out_dir + str("%03d.png" % (i + 1))
        #print(outname)
        copyfile(kf,outname)
    

    min_only = keyframes[::2]
    max_only = keyframes[1::2]
    #print("MIN: ", min_only)
    #print("MAX: ", max_only)

    #make min mean
    kf_min_imgs = []
    for kf in min_only:
        kf_min_imgs.append(cv2.imread(kf,0))
    avgkfmin = np.mean(kf_min_imgs,axis=0)
    outname = out_dir + "min_mean_keyframe.png"
    #print(outname)
    cv2.imwrite(outname,avgkfmin)

    #make min mean
    kf_max_imgs = []
    for kf in max_only:
        kf_max_imgs.append(cv2.imread(kf,0))
    avgkfmax = np.mean(kf_max_imgs,axis=0)
    outname = out_dir + "max_mean_keyframe.png"
    #print(outname)
    cv2.imwrite(outname,avgkfmax)

    #make total mean
    kf_imgs = []
    for kf in keyframes:
        kf_imgs.append(cv2.imread(kf,0))
    avgkf = np.mean(kf_imgs,axis=0)
    outname = out_dir + "mean_keyframe.png"
    #print(outname)
    cv2.imwrite(outname,avgkf)
    

def find_minima(stride_widths,idx):
    """
    Find minima stride width in a sequence.    
    Args:
        stride_widths - width of stride for each frame.
        idx - index of previous minima, 0 for first.
    Return:
       minima + find_maxima(stride_widths[i;],idx+1) - Return minima and 
       recursively call find_maxima function. 
    """
   
    
    #print (len(stride_widths))
    
    #Basecase - if remaining sequence is less than 5 frames return []. 
    if len(stride_widths) <= 5:
        return []

    #Iterate through widths and find minima.
    mn = 10000
    for i,s in enumerate(stride_widths):   
        if s > mn:
            if i > 0:
                minima = [idx +(i-1)]
                mn = 10000
            else:
                continue
        else:
            mn = s
            continue

 #       print ("Min: ", minima[0], stride_widths[i-1])

        try:
            return minima + find_maxima(stride_widths[i:],idx+i)
        except:
            return minima

def find_maxima(stride_widths,idx):
    """
    Find maxima in a sequence.
    Args:
        stride_widths - width of stride for each frame.
    Return:
       maxima + find_minima(stride_widths[i;],idx+1) - Return maxima and 
       recursively call find_minima function. 
   """
    
    #print (len(stride_widths))

    #Basecase -  if remaining sequence is less than 5 frames return [].    
    if len(stride_widths) <= 5:
        return []

    #Iterate through widths and find maxima.
    mx = 0
    for i,s in enumerate(stride_widths):
        if s < mx:
            if i > 0:
                maxima = [idx+(i-1)]#,stride_widths[i-1]]
                mx = 0
            else:
                continue
        else:
            mx = s
            continue
        
  #      print ("Max: ",  maxima, stride_widths[i-1])

        try:
            return maxima + find_minima(stride_widths[i:],idx+i)
        except:
            return maxima

def get_stride_width(frame):
    """
    Find the stride width in pixels.
    Args:
        frame - path of the frame to use.
    Return:
        stride_width - width of stride in pixels.
    """

    img = cv2.imread(str(frame),0)
    height, width = img.shape
    
    #Slightly above bottom to cover all angles 
    bot = height - 30
    left = None
    right = None


    #print ("Height: ", height)
    #print ("Width: ", width)
    #print ("Bottom: ", bot)

    #Find leftmost white pixel
    for p in range(0,width-1):
        if left == None:
            if img.item(bot,p) == 255:
                left = p
    #            print ("Left: ", left)
                break

    #find rightmost white pixel
    for p in range(width-1,0,-1):
        if right == None:
            if img.item(bot,p) == 255:
                right = p
   #             print ("Right: ", right)
                break
    
    #calc width of stride    
    stride_width = right - left

    return stride_width


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Find keyframes from gait silhouette sequences.')
     
    parser.add_argument(
        'base_dir',
        help="Base directory tree containing gait silhouette sequences.")   
    
    args = parser.parse_args()

    main(args.base_dir)
    
