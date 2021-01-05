import xml.etree.ElementTree as ET
import os

def readXML(xmlname, path=None):
    path = os.path.join('..','pics','train_dataset','annotations')
    xml = ET.parse(path + "/" + xmlname)
    filename = xml.find('filename').text
    bndbox_elements = xml.find('object').getchildren()[-1].getchildren()
    bndbox = []
    for element in bndbox_elements:
        bndbox.append(element.text)
    print(filename,bndbox)