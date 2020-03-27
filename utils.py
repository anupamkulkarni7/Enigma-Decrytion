import pickle
import os


def output_file(plain, cipher):
    with open("trial", "w") as f:
        for p, c in zip(plain, cipher):
            if len(p) is not len(c):
                print("Length mismatch")
            f.write(p + '\t' + c + '\n')


def pkl_load(filepath):
    with open(os.path.join(filepath), "rb") as f:
        pdata = pickle.load(f)
    return pdata


def pkl_dump(pdata, filepath):
    with open(os.path.join(filepath), "wb") as f:
        pickle.dump(pdata, f)
