import glob
import os

files = glob.glob("docs/dlomix*.rst")


# Function to replace the first line of each .rst file
def replace_first_line(filename):
    with open(filename, "r+") as file:
        lines = file.readlines()
        if not lines[0].startswith("dlomix"):
            return
        first_word = lines[0].split()[0]
        lines[0] = f"``{first_word}``\n"
        lines[1] = "=" * len(lines[0]) + "\n"
        file.seek(0)
        file.writelines(lines)
        file.truncate()


# Replace the first line for each .rst file
for file in files:
    replace_first_line(file)
