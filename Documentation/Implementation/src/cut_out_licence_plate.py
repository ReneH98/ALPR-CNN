import cv2
import imutils
import numpy as np

"""
@:brief cuts out a licence plate of a picture
@:parameter path: path to a picture
@:parameter DEBUG: if true, prints parameter of masks on console for debug purpose
@:parameter PRINT_ALL_PICS: if true, save all pictures in output folder for debug purpose
@:return picture matrix of licence plate
"""
def getLicencePlate(path, DEBUG = False, PRINT_ALL_PICS = False):
    output_file = "output/" + path.split("/")[0] + "_" + path.split("/")[1].split(".")[0]
    data_type = ".png"
    input_file = "testPics/" + path

    # read image
    img = cv2.imread(input_file)
    # img = cv2.resize(img, (1920,1080) )
    height_original = len(img)
    width_original = len(img[1])

    # convert to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blurr background
    blurr = cv2.bilateralFilter(gray, 11, 35, 17)
    # blurr = cv2.GaussianBlur(gray,(5,5),0)

    # Perform Edge detection
    kernel = np.ones((2, 2), np.uint8)
    edged = cv2.Canny(blurr, 50, 100)
    # edged = cv2.blur(edged, (4, 4))
    edged = cv2.dilate(edged, kernel, iterations=1)
    # edged = cv2.erode(edged,kernel,iterations = 1)
    #edged = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # detect contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    if PRINT_ALL_PICS:
        cv2.imwrite(output_file + "_gray" + data_type, gray)
        cv2.imwrite(output_file + "_blurr" + data_type, blurr)
        cv2.imwrite(output_file + "_edge" + data_type, edged)
        test = cv2.drawContours(img.copy(), cnts, -1, (0, 255, 0), 3)
        cv2.imwrite(output_file + "_contours" + data_type, test)

    i = 1
    possible_candidates = []
    candidate_rating = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.022 * peri, True)
        if len(approx) == 4:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [approx], 0, 255, -1, )
            new_image = cv2.bitwise_and(img, img, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = img[topx:bottomx + 1, topy:bottomy + 1]

            # edit mask which might be licence plate
            thresh = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.bilateralFilter(thresh, 11, 35, 17)
            thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            width = len(thresh[1])
            height = len(thresh)
            ratio = float(width) / height
            width_ratio = float(width) / width_original
            height_ratio = float(height) / height_original

            # cut few percent from left/right, top/bottom
            height_c = int(float(height) * 0.07)
            width_c = int(float(width) * 0.06)
            thresh = thresh[height_c:height - height_c, width_c:width - width_c]

            black_white = thresh.mean()
            # calculate average color from middle row between 15-85% of picture width
            width_thresh = len(thresh[1])
            height_thresh = len(thresh)
            text_color = thresh[int(height_thresh / 2) - 2 : int(height_thresh / 2) + 2, int(width_thresh * 0.15): int(width_thresh * 0.85)].mean()
            # text = pytesseract.image_to_string(thresh, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ --psm 7')

            # create rating: ratio - black/white - width/heigth - detection
            rating = 0
            if ratio >= 3 and ratio <= 4.1:
                rating = rating + 2

            if ratio > 2 and ratio < 3:
                rating = rating + 1

            if ratio > 4.1 and ratio <= 5:
                rating = rating + 1

            if ratio <= 2 or ratio >= 5.5:
                rating = rating - 1

            if text_color >= 150 and text_color <= 180:
                rating = rating + 2

            if text_color > 180 and text_color <= 200:
                rating = rating + 1

            if text_color > 130 and text_color < 150:
                rating = rating + 1

            if text_color < 100 or text_color > 210:
                rating = rating - 1

            if black_white > 130 and black_white < 230:
                rating = rating + 1

            if black_white < 100 or black_white > 240:
                rating = rating - 1

            if width_ratio >= 0.06 and width_ratio <= 0.14:
                rating = rating + 1

            if width_ratio < 0.04 or width_ratio > 0.2:
                rating = rating - 1

            if height_ratio >= 0.025 and height_ratio <= 0.06:
                rating = rating + 1

            if height_ratio < 0.015 or height_ratio > 0.1:
                rating = rating - 1

            if DEBUG:
                print("Mask " + str(i) + ":")
                print("Mean color: " + str(round(black_white,1)))
                print("Text color: " + str(round(text_color,1)))
                print("Height: " + str(round(height,1)))
                print("Width: " + str(round(width,1)))
                print("Height ratio: " + str(round(height_ratio,2)))
                print("Width ratio: " + str(round(width_ratio,2)))
                print("Width/height ratio: " + str(round(ratio,2)))
                print("Rating: " + str(rating) + "\n")
                cv2.imwrite(output_file + "_mask" + str(i) + data_type, thresh)

            possible_candidates.append(cropped)
            candidate_rating.append(rating)
            i = i + 1

    # write to output image
    pos = 0
    for pic in possible_candidates:
        if candidate_rating[pos] == max(candidate_rating):
            cv2.imwrite(output_file + "_LP" + data_type, pic)
            return pic
        pos = pos + 1
