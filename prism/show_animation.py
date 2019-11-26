import cv2
import sys
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    args = sys.argv
    imgs_dir = args[1]
    if not imgs_dir.endswith("/") : imgs_dir += "/"

    img_names_top = [f for f in listdir(imgs_dir+"top_view/")\
        if isfile(join(imgs_dir+"top_view/", f)) and f.endswith(".jpg")]
    img_names_side = [f for f in listdir(imgs_dir+"side_view/")\
        if isfile(join(imgs_dir+"side_view/", f)) and f.endswith(".jpg")]
    img_names_top.sort()
    img_names_side.sort()
    print("Will show", len(img_names_top), len(img_names_side), "frames")
    cv2.namedWindow("top", flags=cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("side", flags=cv2.WINDOW_AUTOSIZE)
    for idx, (img_nt, img_ns) in enumerate(zip(img_names_top, img_names_side)):
        if idx % 100 == 0 : print(f"{idx:n} / {len(img_names_top):n}")
        img_t = cv2.imread(imgs_dir + "top_view/" + img_nt)
        img_s = cv2.imread(imgs_dir + "side_view/" + img_ns)
        cv2.imshow("top", img_t)
        cv2.waitKey(10)
        cv2.imshow("side", img_s)
        cv2.waitKey(10)
    cv2.destroyWindow("top")
    cv2.destroyWindow("side")
