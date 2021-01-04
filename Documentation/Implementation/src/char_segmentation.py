import cv2
import imutils
import numpy as np

"""
@:brief cuts out all characters of a licence plate
@:parameter path: path to a picture
@:parameter DEBUG: if true, prints parameter of masks on console for debug purpose
@:parameter PRINT_ALL_PICS: if true, save all pictures in output folder for debug purpose
@:return array of picture matrices (all chars)
"""
def char_segmentation(path, DEBUG = False, PRINT_ALL_PICS = False):
    output_file = "output/" + path.split("/")[1].split(".")[0]
    data_type = ".png"
    input_file = "testPics/" + path

    # read licence plate
    img = cv2.imread(input_file)
    height_original = len(img)
    width_original = len(img[1])

    # convert to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    blur = cv2.bilateralFilter(gray, 11, 35, 17)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)
    # thresh = cv2.Canny(blur.copy(),50,100)

    # find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    if PRINT_ALL_PICS:
        cv2.imwrite(output_file + "_gray" + data_type, gray)
        cv2.imwrite(output_file + "_blurr" + data_type, blur)
        cv2.imwrite(output_file + "_thresh" + data_type, thresh)
        test = cv2.drawContours(img.copy(), cnts, -1, (0, 255, 0), 3)
        cv2.imwrite(output_file + "_contours" + data_type, test)

    characters = {}
    mask_width = []
    mask_height = []
    mask_ratio = []
    mask_color = []
    char_details = {}
    i = 1
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.022 * peri, True)
        mask = np.zeros(gray.shape, np.uint8)

        new_image = cv2.drawContours(mask, [approx], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        # get some more pixels around the contour
        pixel = round(height_original * 0.05)
        if topx-pixel < 0:
            topx = 0
        else:
            topx = topx - pixel

        if topy-pixel < 0:
            topy = 0
        else:
            topy = topy - pixel

        if bottomx+pixel > len(thresh):
            bottomx = len(thresh)
        else:
            bottomx = bottomx + pixel

        if bottomy+pixel > len(thresh[1]):
            bottomy = len(thresh[1])
        else:
            bottomy = bottomy + pixel

        cropped = img[topx:bottomx, topy:bottomy]

        width = len(cropped[1])
        height = len(cropped)
        ratio = float(width) / height
        width_ratio = float(width) / width_original
        height_ratio = float(height) / height_original

        # get height, width and ratio for median
        HR = round(height_ratio,2)
        WR = round(width_ratio,2)
        R = round(ratio,2)
        C = round(cropped.mean(),2)
        mask_height.append(HR)
        mask_width.append(WR)
        mask_ratio.append(R)
        mask_color.append(C)

        if DEBUG:
            print("Mask " + str(i) + ":")
            print("Height ratio: " + str(HR))
            print("Width ratio: " + str(WR))
            print("Width/height ratio: " + str(R))
            print("Color: " + str(C) + "\n")
            cv2.imwrite(output_file + "_mask" + str(i) + data_type, cropped)

        char_details[topy] = [HR, WR, R, C]
        characters[topy] = cropped
        i = i + 1

    # sort array with data from all masks
    mask_width.sort()
    mask_height.sort()
    mask_ratio.sort()
    mask_color.sort()

    # get median of array + pic before median and average those -> definitely a valid character
    median_width = (np.median(mask_width) + mask_width[int(len(mask_width)/2)-1]) / 2
    median_height = (np.median(mask_height) + mask_height[int(len(mask_height)/2)-1]) / 2
    median_ratio = (np.median(mask_ratio) + mask_ratio[int(len(mask_ratio)/2)-1]) / 2
    median_color = (np.median(mask_color) + mask_color[int(len(mask_color) / 2) - 1]) / 2

    if DEBUG:
        print("Median height: " + str(round(median_height,2)))
        print("Median width: " + str(round(median_width,2)))
        print("Median ratio: " + str(round(median_ratio,2)))
        print("Median color: " + str(round(median_color,2)))

    # sort all masks by their x position in the full pictures = chars from left to right
    import collections
    j = 1
    ret = []
    od = collections.OrderedDict(sorted(characters.items()))
    height_thresh = 0.08
    width_thresh = 0.1
    ratio_thresh = 0.35
    color_thresh = 28
    for pos in od:
        if char_details[pos][0] >= median_height-height_thresh and char_details[pos][0] <= median_height+height_thresh:
            if char_details[pos][1] >= median_width-width_thresh and char_details[pos][1] <= median_width+width_thresh:
                if char_details[pos][2] >= median_ratio-ratio_thresh and char_details[pos][2] <= median_ratio+ratio_thresh:
                    if char_details[pos][3] >= median_color - color_thresh and char_details[pos][3] <= median_color + color_thresh:
                        ret.append(od[pos])
                        if DEBUG:
                            cv2.imwrite(output_file + "_char" + str(j) + data_type, od[pos])
                            j = j + 1

    return ret
