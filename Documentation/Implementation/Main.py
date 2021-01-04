from src import cut_out_licence_plate as COLP
from src import char_segmentation as CS
from src import validate_licence_plate as VLP
from src import OCR
from os import walk

def getAllPlates(folder):
    for (_, _, filenames) in walk("testPics/" + folder):
        filenames.sort()
        for file in filenames:
            if not file.startswith('.') and file.endswith('.png'):
                print("Processing: " + folder + "/" + file)
                COLP.getLicencePlate(folder + "/" + file)

#getAllPlates("Pictures_FH2")

#COLP.getLicencePlate("selfMade/01.png", True)

CS.char_segmentation("LP/23.png", True)

#OCR.readLicencePlate(CS.char_segmentation("LP/23.png"))

