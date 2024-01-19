from medpy import metric
from surface import Surface
import nibabel as nb
import numpy as np
from pathlib import Path
import csv

def get_scores(pred,label,vxlspacing):
	volscores = {}

	volscores['dice'] = metric.dc(pred,label)
	volscores['jaccard'] = metric.binary.jc(pred,label)
	volscores['voe'] = 1. - volscores['jaccard']
	volscores['rvd'] = metric.ravd(label,pred)

	# if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
	# 	volscores['assd'] = 0
	# 	volscores['msd'] = 0
	# else:
	# 	evalsurf = Surface(pred,label, physical_voxel_spacing = vxlspacing, mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
	# 	volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
    #
	# 	volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)

	return volscores


label_path = Path('D:\\Yuliia_data\\Yuliia_data\\lits\\seg\\TEST')
prob_path = Path('D:\\LITS_segmentation\\RESULTS_GABRIELLA\\3D_Ynet\\256_256_64\\inference')


results = []
outpath = 'D:\\LITS_segmentation\\RESULTS_GABRIELLA\\3D_Ynet.csv'

out_path = Path(outpath)

if out_path.exists()==False:
    with open(out_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='|')
        spamwriter.writerow(['case', 'dice liver', 'jaccard liver', 'voe liver', 'rvd liver', 'case', 'dice lesion', 'jaccard lesion', 'voe lesion', 'rvd lesion'])

print('checking code')

for case in prob_path.iterdir():
    print('checking code 2')
    case1 = case.name
    loaded_label = nb.load(label_path/case1)
    loaded_prob = nb.load(prob_path/case1)

    liver_scores = get_scores(loaded_prob.get_data() >= 1, loaded_label.get_data() >= 1,
                              loaded_label.header.get_zooms()[:3])
    lesion_scores = get_scores(loaded_prob.get_data() == 2, loaded_label.get_data() == 2,
                               loaded_label.header.get_zooms()[:3])
    print("Liver dice", liver_scores['dice'], "Lesion dice", lesion_scores['dice'])

    results.append([case1, liver_scores, lesion_scores])

    # create line for csv file
    outstr = str(case1) + ','
    print('checking outstr', outstr)
    for l in [liver_scores, lesion_scores]:
        for k, v in l.items():
            outstr += str(v) + ','
            outstr += '\n'

    # # create header for csv file if necessary
    # if not os.path.isfile(outpath):
    #     headerstr = 'Volume,'
    #     for k, v in liver_scores.iteritems():
    #         headerstr += 'Liver_' + k + ','
    #     for k, v in liver_scores.iteritems():
    #         headerstr += 'Lesion_' + k + ','
    #     headerstr += '\n'
    #     outstr = headerstr + outstr

    # write to file
    print('checking where are we writing metrics', outpath)
    f = open(outpath, 'a+')
    f.write(outstr)
    f.close()
