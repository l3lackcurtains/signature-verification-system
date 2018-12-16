import cv2
import os


def prepare(im_pth, file):
    desired_size = 300

    im = cv2.imread(im_pth, 0)
    old_size = im.shape[:2]

    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [225, 225, 225]
    new_im = cv2.copyMakeBorder(im, top, bottom, left,
                                right, cv2.BORDER_CONSTANT, value=color)
    # TODO: adjust threshold based on signature visibility
    _, thresh1 = cv2.threshold(new_im, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./test/{}'.format(file), thresh1)


def main():
    path = './Dataset/custom'
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        impath = folder.decode('utf-8')+'/'+file.decode('utf-8')
        prepare(impath, file.decode('utf-8'))


main()
