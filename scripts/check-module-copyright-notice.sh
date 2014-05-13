#!/bin/bash

CHEADER="CHEADER.TXT"
PASSED=1

MODULE=$1

# Make sure you are in the script directory
BASEDIR=$(dirname $0)
cd $BASEDIR

# Format C header
rm -rf $CHEADER
echo "/*" > $CHEADER
while read LINE
do
  echo " * $LINE" >> $CHEADER
done < ../HEADER.TXT
echo " */" >> $CHEADER

cd ..
find $MODULE/src -type f > scripts/list.tmp

# Check all the sources
while read FILE
do
  grep "*/" $FILE -m 1 -B 100 | diff - scripts/CHEADER.TXT > /dev/null
  if (( $? == 1 ))
  then
    echo $FILE
    PASSED=0
  fi 
done < scripts/list.tmp

if (( $PASSED == 0 ))
then
  exit  1
fi
