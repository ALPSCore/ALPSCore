#!/usr/bin/python

import os
import sys
import fnmatch
import notice

def writeHeader(f, header):
  for line in header:
    f.write(line)

def fixFile(filename, header):
  with open(filename, "r") as f:
    input = f.readlines()
  with open(filename, "w") as f:
    noticeDone = False
    blockNotice = False
    lineNotice = False
    writeHeader(f, header)
    for line in input:
      if (not noticeDone):
        if (blockNotice):
          if "*/" in line:
            noticeDone = True
        elif (lineNotice):
          if not line.startswith("//"):
            noticeDone = True
            f.write(line)
        else:
          if (line.startswith("//")):
            lineNotice = True
          elif (line.startswith("/*")):
            blockNotice = True
          else:
            noticeDone = True
            f.write(line)
      else:
        f.write(line)

def crawlDirectory(root, filter, header):
  for dirpath, dirnames, filenames in os.walk(root):
    for filename in fnmatch.filter(filenames, filter):
      absFilename = os.path.join(dirpath, filename)
      fixFile(absFilename, header)

def printUsage():
  print("fix-module-copyright-notice.py <moduleName>")
  exit(1)

# Check we have one argument
if len(sys.argv) != 2:
  printUsage()
module = sys.argv[1]

# Make sure we are at the top of the repository
location = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(location)
os.chdir("..")

# Read header
headerFile = "HEADER.TXT"
cFileHeader = notice.readHeader(headerFile)

crawlDirectory(module, "*.cpp", cFileHeader)
crawlDirectory(module, "*.hpp", cFileHeader)
crawlDirectory(module, "*.h", cFileHeader)
crawlDirectory(module, "*.h.in", cFileHeader)


