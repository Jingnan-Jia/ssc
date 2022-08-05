import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    scv_fpath = "results/ellipse_position2.csv"
    plt.figure()
    x_ls, y_ls = [], []
    with open(scv_fpath, 'r') as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            x, y = row
            x_ls.append(int(x))
            y_ls.append(int(y))

            if idx > 5000:
                break

    fig, ax = plt.subplots()
    ax.scatter(x_ls, y_ls, s=1)
    plt.show()
    fig.savefig('ellipse_position4.png')


if __name__ == '__main__':
    main()