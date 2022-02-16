a_file = open("filenames.txt", "r")


string_without_line_breaks = ""

for line in a_file:

  stripped_line = line.strip()

  string_without_line_breaks += stripped_line + " " 

a_file.close()


print(string_without_line_breaks)
