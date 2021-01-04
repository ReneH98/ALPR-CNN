from re import compile

# normal: ["L1ABC", "L12345A"]
# special: ["LA12", "WA12345", "GMHANS1"]
# false: ["L8I8NX", "1ASDF", "L1A"]

"""
@:brief checks if a string can be a licence plate
@:parameter plate: string of a licence plate
@:return true if valid licence plate, else false
"""
def validateLicencePlate(plate):
    plate_format_normal = compile('^[A-Z]{1,3}[0-9]{1,5}[A-z]{1,5}$')
    plate_format_special = compile('^[A-Z]{2,6}[0-9]{1,5}$')
    if plate_format_normal.match(plate) is not None and len(plate) >=4 and len(plate) <= 7:
        return True
    else:
        if plate_format_special.match(plate) is not None and len(plate) >=4 and len(plate) <= 7:
            return True
    return False