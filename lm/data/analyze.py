import os
import csv
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

words = ['before', 'after', 'until', 'since', 'while']

tf = []
tf_dict = {'True' : 1, 'False' : 0}

with open('test.csv', 'r') as f:
	reader = csv.reader(f, delimiter = '\t')
	for row in reader:
		tf.append(tf_dict[row[1]])

files = []
for filename in os.listdir('eval'):
    f = os.path.join('eval', filename)
    # checking if it is a file
    if os.path.isfile(f) and 'generated.output_' in filename and 'last_model' not in filename:
    	files.append(f)

result = []
for file in files:
	model_name = int(file.split('_')[1])
	outputs = {w : [[], []] for w in words}
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter = '\t')
		i = 0
		for row in reader:
			outputs[row[1]][0].append(float(row[2]))
			outputs[row[1]][1].append(tf[i])
			i += 1

	accuracies = dict()
	for w in words:
		x = np.array(outputs[w][0]).reshape(-1, 1)
		y = np.array(outputs[w][1])
		if x.size == 0:
			print(w)
			continue
		model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
		accuracies[w] = classification_report(y, model.predict(x), output_dict = True)['accuracy']

	result.append({model_name : accuracies})

with open('results.csv', 'w') as f:
	writer = csv.writer(f, delimiter = '\t')
	writer.writerow(['data', 'before', 'after', 'until', 'since', 'while'])
	for i in result:
		model_name = list(i.keys())[0]
		x = i[model_name]
		writer.writerow([model_name, x['before'], x['after'], x['until'], x['since'], x['while']])