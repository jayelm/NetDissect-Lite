#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

if [ ! -f dataset/CUB_200/README.txt ]
then

mkdir -p dataset/CUB_200
pushd dataset/CUB_200

echo "Downloading CUB README"
wget --progress=bar \
    http://www.vision.caltech.edu/visipedia-data/CUB-200/README.txt \
    -O README.txt

echo "Downloading CUB images"
wget --progress=bar \
    http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz \
    -O images.tgz
tar -zxf images.tgz
rm images.tgz

echo "Downloading annotations"
wget --progress=bar \
    http://www.vision.caltech.edu/visipedia-data/CUB-200/attributes.tgz \
    -O attributes.tgz
tar -xzf attributes.tgz
rm attributes.tgz

popd

fi
