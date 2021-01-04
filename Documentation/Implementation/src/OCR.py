import cv2
import pytesseract

"""
@:brief converts pictures to string
@:parameter characters: array with pictures matrices (provided as return from char_segmentation.py
@:parameter DEBUG: if true, save all pictures in output folder for debug purpose
@:return licence plate as string
"""
def readLicencePlate(characters, DEBUG = False):
    grays = []
    for img in characters:
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    bilateral = []
    for img in grays:
        bilateral.append(cv2.bilateralFilter(img, 11, 50, 17))

    thresh = []
    for img in bilateral:
        thresh.append(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 20))

    blurr = []
    for img in thresh:
        blurr.append(cv2.blur(img,(2,2)))

    if DEBUG:
        char = 0
        cv2.imwrite("output/gray.png", grays[char])
        cv2.imwrite("output/blur.png", bilateral[char])
        cv2.imwrite("output/thresh.png", thresh[char])
        cv2.imwrite("output/blur2.png", blurr[char])

    text = ""
    for img in thresh:
        text = text + pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ --psm 10')

    print("Detected plate after filters: " + text)