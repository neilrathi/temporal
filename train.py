import LOTlib3
import numpy as np
import random
import pickle

from copy import copy

from LOTlib3.Grammar import Grammar
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib3.DataAndObjects import UtteranceData
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler
from LOTlib3.Miscellaneous import flip

print('=== initializing... ===')
""" DEFINE PCFG """
grammar = Grammar()

# initialize: START → λ A B u . BOOL
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

# intervals: int → a1 / b1 / a2 / b2 / u
grammar.add_rule('int', 'context.A1', None, 1.0)
grammar.add_rule('int', 'context.B1', None, 1.0)
grammar.add_rule('int', 'context.A2', None, 1.0)
grammar.add_rule('int', 'context.B2', None, 1.0)
grammar.add_rule('int', 'context.U', None, 1.0)

""" DEFINE TARGETS """
# let u ∈ [-100, 100]
upper_bound = 100
lower_bound = -100
domain_as_list = range(lower_bound, upper_bound+1)

# w = ⟨before, after, since, until, while⟩
words = ['before', 'after', 'since', 'until', 'while']
uniform_weights = {word: 1 for word in words}

# target meanings m = ⟨m1,..., m5⟩
target_functions = {
    'before': lambda context: lt_(context.A1, context.B1), 
    'after': lambda context: lt_(context.B1, context.A2),
    'since': lambda context: and_(and_(lt_(context.A1, context.U), lte_(context.U, context.A2)), and_(lte_(context.A1, context.B2), lt_(context.B1, context.U))),
    'until': lambda context: and_(and_(lte_(context.A1, context.U), lt_(context.U, context.A2)), and_(lte_(context.B1, context.A2), lt_(context.U, context.B2))),
    'while': lambda context: and_(lt_(context.B1, context.A2), lt_(context.A1, context.B2))
}

""" PRELIMINARY OBJECTS """

# a context consists of two events A, B and an utterance U
class Context(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

class MyHypothesis(LOTHypothesis):
    # inherit from LOTHypothesis
    def __init__(self, grammar = grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar = grammar, maxnodes = 16, display='lambda context: %s', **kwargs)

def make_hypothesis(**kwargs):
    return MyHypothesis(**kwargs)

class TemporalLexicon(SimpleLexicon):
    def __init__(self, value, word_weights = uniform_weights, alpha = 0.95):
        self.value = value
        SimpleLexicon.__init__(self, value=value)
        self.alpha = alpha
        self.word_weights = word_weights

    # evaluates a lexicon on a given context (boolean output)
    def __call__(self, utterance, context):
        return self.value[utterance](context)

    def __hash__(self):
        return hash(str(self))

    # copies to a new temporal lexicon
    def __copy__(self):
        new = type(self)(value = self.value, alpha = self.alpha)
        for w in self.all_words():
            new.set_word(w, copy(self.get_word(w)))

        for k in self.__dict__.keys():
            if k not in ['self', 'value']:
                new.__dict__[k] = copy(self.__dict__[k])

    # partitions a list of utterances into true / false ones for a given context
    def partition_utterances(self, utterances, context):
        trues, falses = [], []
        for u in utterances:
            ret = self(u, context)
            if ret: trues.append(u)
            else: falses.append(u)
        return trues, falses

    # computes p(u | m, c) for an individual utterance u
    def compute_single_likelihood(self, udi):
        assert isinstance(udi, UtteranceData)
        
        u = udi.utterance # extracts utterance from an utterance object
        word_weight = self.word_weights[u]
        trues, falses = self.partition_utterances(udi.possible_utterances, udi.context)
        met_weight = len(trues) + len(falses)
        if u in trues:
            # (α * w)/|true(c)| + (1-α)*(w)/|w|
            p = ((self.alpha * word_weight) / len(trues)) + ((1 - self.alpha) * word_weight) / met_weight
        else:
            # (1-α)*(w)/|w|
            p = ((1 - self.alpha) * word_weight) / met_weight

        return np.log(p)

    # compares two lexicons
    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        for word in self.all_words():
            if self.get_word(word) != other.get_word(word):
                return False
        return True

    # proposes to the lexicon by flipping a coin for each word and proposing to it
    def propose(self):
        fb = 0.0
        changed_any = False

        while not changed_any:
            new = TemporalLexicon({**self.value})

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
def train_data(word, targets, n = 500):
    data = []
    while len(data) < n:
        A = sorted([random.choice(domain_as_list), random.choice(domain_as_list)])
        B = sorted([random.choice(domain_as_list), random.choice(domain_as_list)])

        U = random.choice(domain_as_list)

        context = Context(A1 = A[0], A2 = A[1], B1 = B[0], B2 = B[1], U = U)

        if target_functions[word](context):
            udi = UtteranceData(context = context, utterance = word, possible_utterances = words)
            p = np.exp(targets.compute_single_likelihood(udi))
            if random.random() < p:
                data.append(udi)
    return data

# for each word, produces a list of utterances with weighted length (total = 500)
def make_data(target_lex, option='uniform', n = 100):
    childes = {'before': 7, 'after': 3, 'while': 8, 'since': 2, 'until': 5}
    data = []

    if option == 'childes':
        for word in words:
            ratio = childes[word]
            data += train_data(word, target_lex, n = (n / 5) * ratio)
    else:
        for word in words:
            data += train_data(word, target_lex, n = n)

    return data

# generates utterance weights a la Piantadosi et al. (2012)
def weight(words, nu = 0.25, n = 100000):
    # initialize an empty count dict
    counts = {word : 0 for word in words}

    for i in range(n):
        # generates a random context
        A = sorted([random.choice(domain_as_list), random.choice(domain_as_list)])
        B = sorted([random.choice(domain_as_list), random.choice(domain_as_list)])

        U = random.choice(domain_as_list)

        context = Context(A1 = A[0], A2 = A[1], B1 = B[0], B2 = B[1], D1 = lower_bound, D2 = upper_bound + 1, U = U)

        # evaluates target function on each context
        for word in words:
            if target_functions[word](context):
                counts[word] += 1

    # pt(word) = probability that word is true in an 'average' context
    # weight is inversely proportional to pt with a smoothing term nu
    return {word : (1 / float(nu + counts[word]/float(n))) for word, count in counts.items()}

informativeness_weights = weight(words)
factor = 5.0 / sum(informativeness_weights.values())
for w in informativeness_weights:
    informativeness_weights[w] = informativeness_weights[w] * factor

print('Weighted:', informativeness_weights)

""" MODEL TRAINING """
def increment_model(target_lex, weights, option = 'uniform', max_data = 1000, increment = 10):
    results = []
    total_data = []
    last_hypothesis = None
    iteration = 1
    
    for x in range(0, max_data, increment):
        print(f'training iteration {iteration}...')
        # add incrementally more data
        amount = x + increment

        # set/propose hypothesis
        if last_hypothesis == None:
            h0 = make_hypothesis()
            init = TemporalLexicon({word: h0 for word in words}, word_weights = weights)
        else:
            init = last_hypothesis

        # create new training data
        new_data = make_data(target_lex, n = increment, option = option)
        total_data += new_data

        # run MH for 50k steps and collect top 10 hypotheses
        topn = TopN(N = 10)

        for h in MetropolisHastingsSampler(init, total_data, steps = 50000):
            topn.add(h)

        # get metadata
        target_likelihood = target_lex.compute_likelihood(total_data)
        mean_top_likelihood = np.mean([hyp.compute_likelihood(total_data) for hyp in topn.get_all()])
        likelihood_ratio = round(target_likelihood / mean_top_likelihood, 2)

        print(f'data: {amount}, ratio: {likelihood_ratio}\n')

        result = {'data_amount': amount,
                  'likelihood_ratio': likelihood_ratio,
                  'top_n': topn}

        results.append(result)
        iteration += 1

    return results

def run_model(option = 'childes', weight_option = 'weighted', max_data = 1000, increment = 50, iter_n = 1):
    file = f'data/{option}-{weight_option}/{option}-{max_data}-{increment}-{iter_n}.pkl'

    if weight_option == 'uniform':
        weights = uniform_weights
    elif weight_option == 'weighted':
        weights = informativeness_weights

    target_lex = TemporalLexicon(target_functions, word_weights = weights)
    result = increment_model(target_lex, weights, option = option, max_data = max_data, increment = increment)
    
    with open(file, 'wb') as f:
        pickle.dump(result, f)

    print(f'=== done with iteration {iter_n}! see {file} for data ===')
    return

if __name__ == '__main__':
    for x in range(25):
        run_model(option='childes', weight_option='weighted', max_data = 120, increment = 5, iter_n=x+1)

    print('=== ALL ITERATIONS COMPLETED! ===')
