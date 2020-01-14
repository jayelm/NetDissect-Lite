from settings import OUTPUT_FOLDER
from subprocess import call
from glob import glob

if __name__ == '__main__':
    for ext in ['csv', 'npy', 'mmap']:
        files = glob(f'{OUTPUT_FOLDER}/*.{ext}')
        if files:
            cmd = ['rm', *files]
            print(' '.join(cmd))
            call(cmd)
