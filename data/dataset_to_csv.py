import os
import glob
import pandas as pd
import time
from PIL import Image


def to_csv(root_dir, csv_dir):
    files = glob.glob(root_dir + "/*/*")

    time0 = time.time()
    df = pd.DataFrame()
    print("The number of files: ", len(files))
    for idx, file in enumerate(files):
        if idx % 10000 == 0:
            print("[{}/{}]".format(idx, len(files) - 1))

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))
        df = df.append({'id': face_id, 'name': face_label}, ignore_index=True)

    df = df.sort_values(by=['name', 'id']).reset_index(drop=True)

    df['class'] = pd.factorize(df['name'])[0]

    df.to_csv(csv_dir, index=False)

    elapsed_time = time.time() - time0
    print("elapsted time: ", elapsed_time // 3600, "h", elapsed_time % 3600 // 60, "m")


if __name__ == '__main__':
    to_csv("/data/encryption/liangli/AFD_part2", "./AFD_part2.csv")
