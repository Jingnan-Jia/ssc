import medpy
import medpy.metric
import numpy as np
import seg_metrics.seg_metrics as sg
import SimpleITK as sitk
import time


# shared_folder = "L:\Scratch\Jingnan\lobe_segmentation_examples"  # you can find my prediction and ground truth in this folder

folder = "/data1/jjia/multi_task/mt/scripts/results/lobe/1635031434_34/infer_pred/GLUCOLD"
pred_fpath = folder + "/GLUCOLD_patients_26_seg.nii.gz"
gdth_fpath = "/data1/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/valid/GLUCOLD_patients_26_seg.nii.gz"

gdth_img = sitk.ReadImage(gdth_fpath)
gdth_np = sitk.GetArrayFromImage(gdth_img)
gdth_np[gdth_np!=1] = 0

pred_img = sitk.ReadImage(pred_fpath)
pred_np = sitk.GetArrayFromImage(pred_img)
pred_np[pred_np!=1] = 0

labels = [1]
# pred_np = np.array([[0,0,1],
#                     [0,1,1],
#                     [0,1,1]])
# gdth_np = np.array([[0,0,1],
#                     [0,0,0],
#                     [1,1,1]])
csv_file = 'metrics.csv'
spacing = np.array([1.,2.])

t0 = time.time()
hd = medpy.metric.binary.hd(result=pred_np, reference = gdth_np, voxelspacing=spacing)
hd95 = medpy.metric.binary.hd95(result=pred_np, reference = gdth_np, voxelspacing=spacing)
asd = medpy.metric.binary.asd(result=pred_np, reference = gdth_np, voxelspacing=spacing)
t1 = time.time()

metrics = sg.write_metrics(labels=labels,  # exclude background if needed
                  gdth_img=gdth_np,
                  pred_img=pred_np,
                  csv_file=csv_file,
                           spacing=[1.,2.],
                  metrics=['hd', 'hd95', 'msd'])
t2 = time.time()

print(f"medpy cost {int(t1-t0)} seconds: hd: {hd}, hd95: {hd95}, asd: {asd}")
print("-------------------")
print(f"seg-metrics cost {int(t2-t1)} seconds: {metrics}")