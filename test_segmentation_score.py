import glob
import os
import numpy as np
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import SimpleITK as sitk
import pandas as pd

data_dir = '/home/gabi/Documents/PaperwithYuliia/seg-test-gt/' # Update this path to the folder containing ground truth files
prediction_dir = '/home/gabi/Documents/PaperwithYuliia/seg-pred-vnet'  # Update this path to the folder containing prediction files

# Create the result directory if it doesn't exist
result_dir = '/home/gabi/Documents/PaperwithYuliia/'
os.makedirs(result_dir, exist_ok=True)

metrics = [metric.DiceCoefficient(), metric.JaccardCoefficient(), metric.AverageDistance(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'), metric.VolumeSimilarity()]

labels = {1: 'LIVER',
          2: 'LESION'
          }

evaluator = eval_.SegmentationEvaluator(metrics, labels)

# Get all subject files based on the extension '.nii.gz' in the ground truth directory
subject_files = glob.glob(os.path.join(data_dir, '*.nii.gz'))

for subject_file in subject_files:
    subject_id = os.path.splitext(os.path.basename(subject_file))[0]  # Extract subject ID from file name
    print(f'Evaluating Subject {subject_id}...')

    # Load ground truth image
    ground_truth = sitk.ReadImage(subject_file)

    # Construct the prediction file path using the subject ID
    prediction_file = os.path.join(prediction_dir, f'{subject_id}.gz')

    if os.path.exists(prediction_file):
        prediction = sitk.ReadImage(prediction_file)

        # Evaluate the prediction against the ground truth
        evaluator.evaluate(prediction, ground_truth, f'Subject_{subject_id}')
    else:
        print(f'Prediction file for Subject {subject_id} not found at path: {prediction_file}')

# Write results to CSV files
result_file = os.path.join(result_dir, 'results.csv') # Update this path to the result file
result_summary_file = os.path.join(result_dir, 'results_summary.csv') # Update this path to the result summary file

# Directly pass the file path to CSVWriter without manually opening the file
writer.CSVWriter(result_file).write(evaluator.results)

# Directly pass the file path to CSVStatisticsWriter without manually opening the file
functions = {'MEAN': np.mean, 'STD': np.std}
writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)

print('Evaluations completed.')

# Read the CSV files if needed
results_df = pd.read_csv(result_file, sep=';')
results_summary_df = pd.read_csv(result_summary_file, sep=';')
