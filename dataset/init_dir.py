import os

for i in range(4):
    if not os.path.exists("sub-%d" % i):
        os.mkdir("sub-%d" % i)
    if not os.path.exists("sub-%d/results" % i):
        os.mkdir("sub-%d/results" % i)
