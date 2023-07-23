import sys

J = float(sys.argv[1])
time = float(sys.argv[2])

with open("tests/tomography1.py") as f:
    exec(f.read())

