import argparse
import os
import cv2
from pathlib import Path


def main(base_dir):
    """
    Remove the partial silhouettes that are at the edge of the frame.
    Should be done before alignment.
    Args: 
        base_dir - Directory tree of silhouette sequences.
    """
    indir = Path(base_dir)

    subjects = list(indir.glob("*"))
   
    lowest_dirs = []
    
    #get all folders containing sil sequences 
    for root,dirs,files in os.walk(indir):
        if files and not dirs:
            lowest_dirs.append(root)
            
    lowest_dirs.sort()
    #print ("lowest_dirs: ", lowest_dirs)

    edge_sils = []
    #Find edge sils
    for d in lowest_dirs:
        print ("Directory: ", d)
        edge_sils += find_edge_sils(d)
        #print(edge_sils) 
    
    
    #delete edge sils
    for sil in edge_sils:
        #print (sil)
        if os.path.exists(sil):
            os.remove(sil)
    
    print (str(len(edge_sils)) + " images removed.")

def find_edge_sils(sil_dir):
    """
    Find the silhouettes at the edge of the frame.
    Args:
        sil_dir - Directory of silhouettes.
    Return: edge_sils - list of file paths of edge silhouettes.
    """
    
    #gap between edge of image and silhouette.
    gap = 10
 
    edge_sils = []
    for sil in Path(sil_dir).glob("*"):
        #print(sil)
        try:
            img = cv2.imread(str(sil),0)   
            height,width = img.shape
        except:
            continue
        #find any white pixels in far left column of pixels
        #(pixel idx 1 is used instead of 0 for width as the left edge on 
        #the sil data has a 1 pixel gap for some reason.)
        for p in range(0,height-1):
            if img.item(p,gap) == 255:
                #print ("Left Edge: ", sil)
                edge_sils.append(sil)
                break

        #find any white pixel in far right column of pixels
        for p in range(0,height-1):
            if img.item(p,(width-1)-gap) == 255:
                #print ("Right Edge: ", sil)
                edge_sils.append(sil)
                break

        #Only look at top and bottom before alignments.
        #Doing so after aligment will remove all frames.

        #find any white pixel in top row of pixels
        for p in range(0,width-1):
            if img.item(gap,p) == 255:
                #print ("Top Edge: ", sil)
                edge_sils.append(sil)
                break
        
        #find any white pixel in bottom row of pixels
        for p in range(0,width-1):
            if img.item((height-1)-gap,p) == 255:
                #print ("Bottom Edge: ", sil)
                edge_sils.append(sil)
                break
        

    return edge_sils

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Find and remove silhouettes at the edge of a frame(Do before alignment).')
     
    parser.add_argument(
        'base_dir',
        help="Base directory tree containing gait silhouette sequences.")   
   
    args = parser.parse_args()

    main(args.base_dir)
 
