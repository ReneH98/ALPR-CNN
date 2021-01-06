import os,re

filelist = os.listdir('data/')

result = [[e for e in re.split("[^0-9]", filename) if e != ''] for filename in filelist]
result = result[1:]
result = [int(e[0]) for e in result if e != '']

maxcount = max(result)

print("Last element in the folder data/ has number " + str(maxcount) + ".\nIs that correct? (y/n)")
val = input("> ")

if val == "y":
    pass
elif val == "n":
    print("What number is it instead? ")
    maxcount = int(input("> "))

print("Copy the new data into the folder now. Once coping is finished hit Enter.")
input("> ")

# start renaming
newfilelist = os.listdir('data/')

to_rename = [e for e in newfilelist if e not in filelist]
to_rename = [e for e in to_rename if ".png" in e]

maxcount += 1
for filename in to_rename:
    maxcount += 1
    helper = filename.split(".")
    fname = helper[0]
    fileending = helper[-1]
    #rename the png file
    os.rename("data/" + fname + ".png", "data/Cars" + str(maxcount) + ".png")
    print(fname + ".png -> " + "Cars" + str(maxcount) + ".png")
    #rename the corresponding xml file
    os.rename("data/" + fname + ".xml", "data/Cars" + str(maxcount) + ".xml")
    print(fname + ".xml -> " + "Cars" + str(maxcount) + ".xml")

