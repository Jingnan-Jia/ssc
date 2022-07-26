import numpy as np
import SimpleITK as sitk
import pandas as pd
from glob import glob


pat_id_xlsx = "/data/jjia/ssc_scoring/ssc_scoring/dataset/ScoringPerSlice.xlsx"
pat_id_df = pd.read_excel(pat_id_xlsx, engine='openpyxl')
reader = sitk.ImageSeriesReader()

for index, row in pat_id_df.iterrows():
    path = row['path']
    file = glob
    reader.SetFileNames(path)
    image = reader.Execute()
    print(row)