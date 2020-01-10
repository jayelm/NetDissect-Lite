#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

if [ ! -f dataset/CUB_200_2011/README ]
then

echo "Downloading CUB dataset"
mkdir -p dataset
pushd dataset
wget --progress=bar \
    http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz \
    -O CUB_200_2011.tgz
tar -zxf CUB_200_2011.tgz
rm CUB_200_2011.tgz
popd

fi

if [ ! -f dataset/CUB_200_2011/images/001.Black_footed_Albatross/img.npz ]
then

echo "Saving to .npz"
python script/save_cub_np.py

fi
