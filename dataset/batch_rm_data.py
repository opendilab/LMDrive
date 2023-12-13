import sys
import os

name = sys.argv[1]
for i in range(14):
    subs = os.listdir("weather-%d/data" % i)
    for sub in subs:
        if name not in sub:
            continue
        os.system("rm -rf " + os.path.join("weather-%d" % i, "data", sub))
        print("delete", os.path.join("weather-%d" % i, "data", sub))
