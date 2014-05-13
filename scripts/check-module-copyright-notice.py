#!/usr/bin/python

import sys
import os
import fnmatch

import notice

# Matches the first lines of the file to the header
def headerMatches(f, header):
  for line in header:
    fileLine = f.readline()
    if (line != fileLine):
      return False
  return True

def crawlDirectory(root, filter, header):
  matches = True
  for dirpath, dirnames, filenames in os.walk(root):
    for filename in fnmatch.filter(filenames, filter):
      absFilename = os.path.join(dirpath, filename)
      with open(absFilename) as f:
        if (not headerMatches(f, header)):
          print absFilename
          global matchSuccesful
          matches = False
  return matches

def printUsage():
  print("check-module-copyright-notice.py <moduleName>")
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

# Crawl all cpp, hpp and h files
# Be careful to avoid the optimization of logical expressions
matches = crawlDirectory(module, "*.cpp", cFileHeader)
matches = crawlDirectory(module, "*.hpp", cFileHeader) and matches
matches = crawlDirectory(module, "*.h", cFileHeader) and matches
matches = crawlDirectory(module, "*.h.in", cFileHeader) and matches

if (not matches):
  exit(1)
