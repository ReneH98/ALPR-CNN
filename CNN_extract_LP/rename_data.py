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

for filename in to_rename:
    maxcount += 1
    fileending = filename.split(".")[-1]
    os.rename("data/" + filename,"data/Cars" + str(maxcount) + "." + fileending)

