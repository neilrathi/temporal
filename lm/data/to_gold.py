import csv

text = []
index = []

with open('test.csv', 'r') as f:
	file = csv.reader(f, delimiter = '\t')
	for line in file:
		text.append(line[2])
		index.append(line[2].split(' ').index(line[0]))

with open('sample/generated.eval', 'w') as f:
    for line in index:
        f.write(f'{line}\n')

with open('sample/generated.text', 'w') as f:
    for line in text:
        f.write(f'{line}\n')