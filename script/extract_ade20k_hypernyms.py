from scipy.io import loadmat

if __name__ == '__main__':
    MAT_FILE = 'dataset/ADE20K_2016_07_26/index_ade20k.mat'

    mat = loadmat(MAT_FILE)
    hypernyms = mat['index'][0][0][11][0]
    breakpoint()
    print(mat['index'][0][0][4])
    print(mat['index'][0][0][5])
    print(mat['index'][0][0][6])
    print(mat['index'][0][0][7])
    print(mat['index'][0][0][8])
    print(mat['index'][0][0][9])

    for row in hypernyms:
        row = row[0]
        if row:
            splits = row.split('. ')
            syns = [s.split(', ') for s in splits]
            syns = [s for s in syns if s != '']
            print(syns)
