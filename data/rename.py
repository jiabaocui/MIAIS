import os

data_path = '/data/encryption/liangli/AFD_part2'
folders = os.listdir(data_path)

# for i, folder in enumerate(folders):
#     files_path = os.path.join(data_path, folder)
#     files = os.listdir(files_path)
#     print(8 * '*')
#     for j, file in enumerate(files):
#         if file[0] != 'f':
#             print(os.path.join(files_path, 'face_' + file))
#             os.rename(os.path.join(files_path, file), os.path.join(files_path, 'face_' + file))
#         else:
#             print('pass')
