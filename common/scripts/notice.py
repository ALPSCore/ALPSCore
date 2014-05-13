# Reades the header from the header file
def readHeader(headerFile):
  cFileHeader = ["/*\n"]
  with open(headerFile) as f:
    for line in f:
      cFileHeader.append(" * " + line)
  cFileHeader.append(" */\n")
  return cFileHeader
