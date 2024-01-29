import cv2 
import numpy as np 
import argparse 
import pdb 

def load_img(path):
    # read bw image
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def vis_compare(img1, img2):
    # get pixel-wise similarity of two images
    # ignoring all white pixels 
    # return: float

    if img1 is None or img2 is None:
        # some program failed, return 0
        return 0.0

    # img1[img1 < 255] = 0
    # img2[img2 < 255] = 0

    # get all pixel indices less than 255
    ones = np.ones(img1.shape)
    ones1 = ones.copy()
    ones1[img1 == 255] = 0
    ones2 = ones.copy()
    ones2[img2 == 255] = 0

    # pdb.set_trace()

    idxs1 = set(np.flatnonzero(ones1).tolist()) 
    idxs2 = set(np.flatnonzero(ones2).tolist())

   
    match_pixels = np.equal(img1, img2)
    # only check if non-zero pixels match

    # intersect
    pixels_nonzero_both = idxs1 | idxs2


    match_pixels = match_pixels.reshape(-1)[np.array(list(pixels_nonzero_both), dtype=np.int64)]
    num_matches = np.sum(match_pixels)

    return num_matches / len(pixels_nonzero_both)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    args = parser.parse_args()
    img1 = load_img(args.img1)
    img2 = load_img(args.img2)
    print(vis_compare(img1, img2))