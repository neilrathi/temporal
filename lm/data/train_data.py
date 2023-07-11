import random

raw = 'raw.txt'
out = 'cleaned.txt'

connectives = ['before', 'after', 'until', 'since', 'while']
connectives_dict = {x : [] for x in connectives}
outputs = set()

with open(raw, 'r') as f:
    for line in f:
        for w in connectives:
            if w in line:
                connectives_dict[w].append(line)

min_len = len(min(connectives_dict.values(), key=len))

for w in connectives_dict:
    outputs.update(random.sample(connectives_dict[w], min_len))

with open(out, 'w') as f:
	for line in outputs:
		f.write(f'{line}')