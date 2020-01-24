#!/usr/bin/env python3

# Tense
#
# Author: Geoff Bacon <bacon@berkeley.edu>
# URL: <url>

"""
Code for ``Tense systems across languages support efficient communication'' by
Geoff Bacon, Yang Xu & Terry Regier.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from partitionsets.partition import Partition
from collections import Counter

# == Utilities == #

def _has_category(c):
    '''
    Return 1 if cell c indicates any category, otherwise 0.

    Inflectional and periphrastic morphemes were coded as such, but for the
    time being we want to ignore that detail.
    '''
    if c and not c.isdigit() and c is not 'n' :
        return 1
    return 0

def _find_category(i, system):
    '''A helper function to find location i's category in a system.'''
    for category in system:
        if i in category:
            return category
    return 'Not found'

def myplot(X, Y, A, B):
    """Convenience function for plotting."""
    plt.figure(figsize=(8,6), dpi=80)
    ax = plt.subplot(111)
    plt.scatter(X, Y, linewidth=5.0, marker='o', color='r')
    plt.scatter(A, B, linewidth=0.1, marker='o', color='grey')

    plt.xlim(0, A.max()*1.1)
    plt.xticks(list(set(A)))
    plt.xlabel('Complexity')

    plt.ylim(0, B.max()*1.1)
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    plt.ylabel('Communicative cost')

    plt.legend(loc='upper right', frameon=False)
    plt.title('Efficiency analysis of tense systems')

    systems = Counter(zip(X, Y))
    for system in systems:
        if systems[system] >= 3:
            plt.annotate(str(systems[system]), xy = system, xytext = (system[0]+0.2, system[1]))

    plt.savefig('tense_systems_wals222.png',dpi=72)
    plt.show()

# == Timeline == #

timeline = list(range(7))
t_0 = 3 # index of the present, used in calculations of similarity to the present.

# Keys are 0-indexed column numbers in (shifted) csv file, values are indexes of timeline
# e.g. `2: [4,5,6]` means the third column in the csv file corresponds to the category
# grouping indexes 4, 5 and 6 of the timeline together (the future).
extensions = {0: [0,1,2], 1: [3], 2: [4,5,6], 3: [2], 4: [1], 5: [0], 8: [4],
              9: [5, 6], 10: [3,4,5,6], 11: [0,1,2,3]}

# need probs estimated from google
google = [0.1034, 0.0795, 0.1839, 0.6183, 0.0074, 0.0048, 0.0028]
#uniform = {i:1/len(timeline) for i in timeline}
need_probability = {i:google[i] for i in timeline}


# == Language == #

class Language():
    def __init__(self, row):
        self.row = row
        self.code = row[1]
        self.name = row[2]
        self._build_system()
        self.simplicity = len(self.system)
        self._calculate_expected_cost()

    def _build_system(self):
        '''
        Turn one row from csv file into a tense system.

        A tense system is a list of lists of timeline indexes
        e.g. English = [[0,1,2], [3], [4,5,6]]
        '''
        system = []
        categories = self.row[10:22]
        for count, cat in enumerate(categories):
            value = _has_category(cat)
            if value:
                if count not in [6, 7]:
                    system.append(extensions[count])
        existing_categories = [item for sublist in system for item in sublist] # flatten list
        unmarked = [i for i in timeline if i not in existing_categories]
        if unmarked:
            system.append(unmarked)
        self.system = system

    def _calculate_expected_cost(self):
        self.informativeness = np.sum([self._communicative_cost(i) * need_probability[i] for i in timeline])

    def _communicative_cost(self, i):
        return np.log2(1 / self._listeners_interpretation(i))

    def _listeners_interpretation(self, i):
        category = self._find_category(i)
        extension = np.sum([self._f(location, category) for location in category])
        return self._f(i, category) / extension

    def _f(self, i, c):
        if i in c:
            return self._sim(i, t_0)
        return 0

    def _sim(self, i,j):
        return np.exp(-1.0 * self._dist(i,j))

    def _dist(self, i,j):
        return abs(timeline[j] - timeline[i])

    def _find_category(self, i):
        '''A helper function to find location i's category in a  system.'''
        for category in self.system:
            if i in category:
                return category
        return 'Not found'

# == Hypothetical == #

class Hypothetical():
    def __init__(self, system):
        self.row = row
        self.system = system
        self.simplicity = len(self.system)
        self._calculate_expected_cost()

    def _calculate_expected_cost(self):
        self.informativeness = np.sum([self._communicative_cost(i) * need_probability[i] for i in timeline])

    def _communicative_cost(self, i):
        return np.log2(1 / self._listeners_interpretation(i))

    def _listeners_interpretation(self, i):
        category = self._find_category(i)
        extension = np.sum([self._f(location, category) for location in category])
        return self._f(i, category) / extension

    def _f(self, i, c):
        if i in c:
            return self._sim(i, t_0)
        return 0

    def _sim(self, i,j):
        return np.exp(-1.0 * self._dist(i,j))

    def _dist(self, i,j):
        return abs(timeline[j] - timeline[i])

    def _find_category(self, i):
        '''A helper function to find location i's category in a  system.'''
        for category in self.system:
            if i in category:
                return category
        return 'Not found'

# == Read in data == #

fname = 'wals_tense_data_cats_shifted.csv'
languages = []
with open(fname, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        lg = Language(row)
        languages.append(lg)

# == Generate hypothetical systems == #

hypotheticals = []
for p in Partition(timeline):
    hyp = Hypothetical(p)
    hypotheticals.append(hyp)


# == Plot results == #

attested_simplicity = np.array([lg.simplicity for lg in languages])
attested_informativeness = np.array([lg.informativeness for lg in languages])
hypothetical_simplicity = np.array([hy.simplicity for hy in hypotheticals])
hypothetical_informativeness = np.array([hy.informativeness for hy in hypotheticals])
myplot(attested_simplicity, attested_informativeness, hypothetical_simplicity, hypothetical_informativeness)
