import pyscreenshot as ImageGrab

# part of the screen
im=ImageGrab.grab(bbox=(65,325,980,850))

# to file
# im.save('test.jpg','JPEG')
im.show()