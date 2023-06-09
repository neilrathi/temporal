import LOTlib3
import numpy as np
import random
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
import scipy.stats

from copy import copy

from LOTlib3.Grammar import Grammar
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib3.DataAndObjects import UtteranceData
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler
from LOTlib3.Miscellaneous import flip

""" DEFINE PCFG """
grammar = Grammar()

# initialize: START → λ A B t . BOOL
grammar.add_rule('START', '', ['BOOL'], 1.0)

# terminations: BOOL → true / false
grammar.add_rule('BOOL', 'True', None, 1.0)
grammar.add_rule('BOOL', 'False', None, 1.0)

# connectives: BOOL → (and BOOL BOOL) / (or BOOL BOOL) / (not BOOL)
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

# comparatives: BOOL → (= int int) / (< int int) / (≤ int int)
grammar.add_rule('BOOL', 'eq_', ['int', 'int'], 1.00)
grammar.add_rule('BOOL', 'lt_', ['int', 'int'], 1.00)
grammar.add_rule('BOOL', 'lte_', ['int', 'int'], 1.00)

# intervals: int → a1 / b1 / a2 / b2 / t
grammar.add_rule('int', 'context.A1', None, 1.0)
grammar.add_rule('int', 'context.B1', None, 1.0)
grammar.add_rule('int', 'context.A2', None, 1.0)
grammar.add_rule('int', 'context.B2', None, 1.0)
grammar.add_rule('int', 'context.U', None, 1.0)

""" DEFINE TARGETS """
# let t ∈ [-100, 100]
upper_bound = 100
lower_bound = -100
domain_as_list = range(lower_bound, upper_bound+1)

# w = ⟨before, after, since, until, while⟩
words = ['before', 'after', 'since', 'until', 'while']
uniform_weights = {word: 1 for word in words}

# m = ⟨m1,..., m5⟩
target_functions = {
    'before': lambda context: lt_(context.A1, context.B1), 
    'after': lambda context: lt_(context.B1, context.A2),
    'since': lambda context: and_(and_(lt_(context.A1, context.U), lte_(context.U, context.A2)), and_(lte_(context.A1, context.B2), lt_(context.B1, context.U))),
    'until': lambda context: and_(and_(lte_(context.A1, context.U), lt_(context.U, context.A2)), and_(lte_(context.B1, context.A2), lt_(context.U, context.B2))),
    'while': lambda context: and_(lt_(context.B1, context.A2), lt_(context.A1, context.B2))
}

# define a context object
class Context(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

# define a hypothesis object
class MyHypothesis(LOTHypothesis):
    # inherit from LOTHypothesis
    def __init__(self, grammar = grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar = grammar, maxnodes = 16, display='lambda context: %s', **kwargs)

def make_hypothesis(**kwargs):
    return MyHypothesis(**kwargs)

# define a TemporalLexicon object
class TemporalLexicon(SimpleLexicon):
    def __init__(self, value, word_weights = uniform_weights, alpha = 0.95):
        # SimpleLexicon requires a self.value attribute, but we pass value as an input to TempLex
        self.value = value
        SimpleLexicon.__init__(self, value=value)
        self.alpha = alpha
        self.word_weights = word_weights

    # evaluate a lexicon on a context (returns True/False)
    def __call__(self, utterance, context):
        return self.value[utterance](context)

    def __hash__(self):
        return hash(str(self))

    def __copy__(self):
        new = type(self)(value = self.value, alpha = self.alpha)
        for w in self.all_words():
            new.set_word(w, copy(self.get_word(w)))

        for k in self.__dict__.keys():
            if k not in ['self', 'value']:
                new.__dict__[k] = copy(self.__dict__[k])

    def partition_utterances(self, utterances, context):
        """
        Take some utterances and a context. Returns 2 lists, giving those utterances
        that are true/false.
        """
        trues, falses = [], []
        for u in utterances:
            ret = self(u,context)
            if ret: trues.append(u)
            else: falses.append(u)
        return trues, falses

    def compute_single_likelihood(self, udi):
        assert isinstance(udi, UtteranceData)
        
        u = udi.utterance
        word_weight = self.word_weights[u]
        trues, falses = self.partition_utterances(udi.possible_utterances, udi.context)
        met_weight = len(trues) + len(falses)
        if u in trues:
            p = ((self.alpha * word_weight) / len(trues)) + ((1 - self.alpha) * word_weight) / met_weight
        else:
            p = ((1 - self.alpha) * word_weight) / met_weight

        return np.log(p)

    def __eq__(self, other):
        """ Compare hypotheses for each word in lexicon
        """
        if not isinstance(other, self.__class__): return False
        for word in self.all_words():
            if self.get_word(word) != other.get_word(word):
                return False
        return True

    def propose(self):
        """
        Propose to the lexicon by flipping a coin for each word and proposing to it.

        This permits ProposalFailExceptions on individual words, but does not return a lexicon
        unless we can propose to something.
        """


        fb = 0.0
        changed_any = False

        while not changed_any:
            new = TemporalLexicon({**self.value}) ## Now we just copy the whole thing

            for w in self.all_words():
                    if flip(self.propose_p):
                        try:
                            xp, xfb = self.value[w].propose()

                            changed_any = True
                            new.value[w] = xp
                            fb += xfb

                        except ProposalFailedException:
                            pass


            return new, fb

""" DATA GENERATION """
# creates a list of n (true) utterances for word n
def test_data(targets, words = words, n = 5):
    data = {x: {} for x in words}
    for word in data:
        true_count = 0
        false_count = 0
        while true_count + false_count < 2 * n:
            A = sorted([random.choice(domain_as_list), random.choice(domain_as_list)])
            B = sorted([random.choice(domain_as_list), random.choice(domain_as_list)])

            U = random.choice(domain_as_list)

            context = Context(A1=A[0], A2=A[1], B1=B[0], B2=B[1], U=U)
            
            udi = UtteranceData(context = context, utterance = word, possible_utterances = words)
            p = np.exp(targets.compute_single_likelihood(udi))

            if random.random() < p:
                target_value = target_functions[word](context)

                if target_value == True and true_count < n:
                    data[word][context] = True
                    true_count += 1
                elif target_value == False and false_count < n:
                    data[word][context] = False
                    false_count += 1
    return data

print('generating data...')

target_lex = TemporalLexicon(target_functions, word_weights = uniform_weights)

test = test_data(target_lex)

files = []
for filename in os.listdir('data/uniform-uniform'):
    f = os.path.join('data/uniform-uniform', filename)
    # checking if it is a file
    if os.path.isfile(f) and '.pkl' in filename:
        files.append(f)

def f1(tp, fp, fn):
    return tp / (tp + 0.5*(fp + fn))

accuracy_list = []

print('testing models...')

for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)

    model_accuracy = {x : [] for x in words}

    for epoch in data:
        ct = {x : 0 for x in words}
        tp = {x : 0 for x in words}
        fp = {x : 0 for x in words}
        fn = {x : 0 for x in words}
        topn = epoch['top_n'].get_all(sorted=True)
        for word in test:
            for lexicon in topn:
                for context in test[word]:
                    correct_value = test[word][context]
                    test_value = lexicon(word, context)
                    if (test_value == correct_value) and (test_value == True):
                        tp[word] += 1
                    elif (test_value != correct_value) and (test_value == True):
                        fp[word] += 1
                    elif (test_value != correct_value) and (test_value == False):
                        fn[word] += 1
            model_accuracy[word].append(tp[word] / (tp[word] + 0.5*(fp[word] + fn[word])))
    accuracy_list.append(model_accuracy)

accuracy = {word : [] for word in accuracy_list[0].keys()}

num_models = len(accuracy_list)

for model in accuracy_list:
    for word in accuracy:
        accuracy[word].append(model[word])

def find_confidence_interval(data, confidence = 0.95):
    a = 1.0 * np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a)-1)
    return m, m-h, m+h

def mean_confidence_interval(data, confidence=0.95):
    return [find_confidence_interval(i) for i in list(zip(*data))]

confidence_intervals = {word : mean_confidence_interval(accuracy[word]) for word in accuracy}

with open('results/before-after-LOT.csv', 'w') as outfile:
    writer = csv.writer(outfile, delimiter = '\t')
    writer.writerow(['data', 'word', 'mean', 'low', 'high'])
    for word in confidence_intervals:
        i = 0
        x = range(0, 650, 25)
        for y in confidence_intervals[word]:
            writer.writerow([x[i], word, y[0], y[1], y[2]])
            i += 1

print('done')