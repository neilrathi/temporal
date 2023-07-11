import os
import csv
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

words = ['before', 'after', 'until', 'since', 'while']

tf = []

with open('test.csv', 'r') as f:
	reader = csv.reader(f, delimiter = '\t')
	for row in reader:
		tf.append(row[1])

files = []
for filename in os.listdir('eval'):
    f = os.path.join('eval', filename)
    # checking if it is a file
    if os.path.isfile(f) and 'generated.output_' in filename and 'last_model' not in filename:
    	files.append(f)

result = []
for file in files:
	model_name = int(file.split('_')[1])
	outputs = {w : {'True' : [], 'False' : []} for w in words}
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter = '\t')
		i = 0
		for row in reader:
			outputs[row[1]][tf[i]].append(float(row[2]))
			i += 1

	accuracies = dict()
	for w in words:
		accuracies[w] = mean(outputs[w]['True']) - mean(outputs[w]['False'])

	result.append({model_name : accuracies})

with open('results.csv', 'w') as f:
	writer = csv.writer(f, delimiter = '\t')
	writer.writerow(['data', 'before', 'after', 'until', 'since', 'while'])
	for i in result:
		model_name = list(i.keys())[0]
		x = i[model_name]
		writer.writerow([model_name, x['before'], x['after'], x['until'], x['since'], x['while']])