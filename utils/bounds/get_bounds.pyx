import cv2

#cpdef unsigned char[:, :] bounds(unsigned char [:, :] image):
cpdef bounds(unsigned char [:, :] image):
    """
    Get the top and bottom bounds of a silhouette.
    """
    #set variable extension types    
    cdef int w, h, x, y, x1, y1

    h = image.shape[0]
    w = image.shape[1]

    #get top pixel
    for y in range(0, h):
        for x in range(0, w):
            if image[y,x] == 255:
                top = [x,y]
                break
            else:
                continue
            #break

    #find bottom pixel
    for y1 in range(h-1, 0, -1):
        for x1 in range(w-1, 0, -1):
            if image[y1,x1] == 255:
                bottom = [x1,y1]
                break
            else:
                continue
            #break

    return top, bottom
