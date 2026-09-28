with open("vector_store_v2.py", "r") as f:
    lines = f.readlines()

with open("vector_store_v2.py", "w") as f:
    for line in lines:
        if "import psutil" in line:
            continue
        if "rss =" in line and "psutil" in line:
            f.write("        rss = 0  # psutil removed as per instructions\n")
        else:
            f.write(line)
