"""
Verifies that your submission zip has the correct structure and contains all the
needed files.

    python checkA4Submission.py path_to_your_andrewid.zip

checks your zip for the correct structure

1. create a folder called <andrewid>
2. put your python files in your <andrewid> folder
3. create your zip file called <andrewid.zip>
4. run this script.

This is *not* a correctness check

Written by Chen Kong, 2018.
"""

import sys, os.path
from zipfile import ZipFile as ZipFile
zip_path = sys.argv[1]

if os.path.isfile(zip_path):
    print('{} is found.'.format(zip_path))
else:
    print('Could not find handin zip.')

andrewid = os.path.basename(zip_path).split('.')[0]
print('Your Andrew Id is {}.'.format(andrewid))

with ZipFile(zip_path, 'r') as zip:
    filelist = zip.namelist()

correct_files = [
    andrewid+'_hw4.pdf',
    'q2_1_eightpoint.py',
    'q2_2_sevenpoint.py',
    'q3_1_essential_matrix.py',
    'q3_2_triangulate.py',
    'q4_1_epipolar_correspondence.py',
    'q4_2_visualize.py',
    'q5_bundle_adjustment.py',
    'q6_ec_multiview_reconstruction.py',
    'q2_1.npz',
    'q2_2.npz',
    'q3_1.npz',
    'q3_3.npz',
    'q4_1.npz',
    'q4_2.npz',
    'q6_1.npz',
]

correct = True
for f in correct_files:
    if os.path.join(andrewid, f) not in filelist and f not in filelist:
        print('{} is not found.'.format(os.path.join(andrewid, f)))
        correct = False

if correct:
    print('Your submission looks good!')
