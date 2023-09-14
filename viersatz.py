#!/bin/python3

import numpy as np
from musictree import *
from pathlib import Path
import argparse

'''
bar=(grundton, 1, 2, 3, 4)
'''

total_octaves = 3
midi_offset = 48
size = total_octaves*12
'''0,2,4,5,7,9,11,12'''

base_leiter = (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1)
base_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]
notes = np.array([note+str(i) for i in range(total_octaves)
                 for note in base_notes])
leiter = np.array(total_octaves*base_leiter)
accord_helper = np.array(2*total_octaves*(1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0))
sept_accord_helper = np.array(
    2*total_octaves*(1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1))

# jumps = (0,2,3,4,5,7,8,9,10,12)
jumps = (1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1)
all_jumps = np.array(2*total_octaves*jumps)

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lily", action="store_true", help="Print a flower before generating music")

args = parser.parse_args()

if args.lily:
    print("""
               __/)
            .-(__(=:
            |  _ \)
      (\__  | / \\
     :=)__)-|/ __/)
      (/    |-(__(=:
         _  |  _ \)
        ( \ | / )
        __ \|/
       (  \ |  _
           \| / )
            |/
            |""")


def choice(a, size=None, replace=True, p=None):

    if p is not None:
        p = p/sum(p)
    return np.random.choice(a=a, size=size, replace=replace, p=p)


def random(p):

    return np.random.random() < p


def index_to_note(index):

    return notes[index]


def vector_to_note(vector):

    return notes[np.nonzero(vector)]


def up(note):

    if (2*base_leiter)[note % 12+1]:
        return note+1
    else:
        return note+2


def down(note):

    if (2*base_leiter)[note % 12-1]:
        return note-1
    else:
        return note-2


def towards(note, nextnote):

    if note > nextnote:
        return down(note)
    else:
        return up(note)


def legal(ton, a, used_jumps):

    all_jumps_index = size - ton

    jump_modifier = np.zeros(size, dtype=int)
    jump_modifier[max(0, ton-12):min(size, ton+12)] = 1

    non_parallel = np.ones(size, dtype=int)
    for j in used_jumps:
        if ton+j >= 0 and ton+j < size:
            non_parallel[ton+j] = 0

    return leiter *\
        all_jumps[all_jumps_index:size+all_jumps_index] *\
        jump_modifier *\
        non_parallel *\
        a


def accord(grundton):

    if grundton >= 0:
        return leiter *\
            accord_helper[size-grundton:2*size-grundton]
    else:
        return leiter *\
            sept_accord_helper[size+grundton:2*size+grundton]


def next(prev, grundton, start=None, temperature=100):

    bar = -1*np.ones(5, dtype=int)
    bar[0] = grundton

    a = accord(grundton)

    used_jumps = []
    used_notes = np.ones(size)
    for i in [4, 1, 3, 2]:

        # legality
        p = legal(prev[i], a, used_jumps=used_jumps)

        # move away from starting note/temperature
        if start is not None:
            temp_modifier = np.empty(size)
            for k in range(size):
                temp_modifier[k] = np.exp(-np.abs(k -
                                          start[i])/6*100/temperature)
            # print(temp_modifier)
            # input()
            p = p*temp_modifier

        # lower than base/higher than soprano
        part_modifier = np.ones(size)
        if bar[4]:
            part_modifier[:bar[4]] = 0.01
        if bar[1]:
            part_modifier[bar[1]:] = 0.1

        p = p*part_modifier

        # used notes modifier

        p = p*used_notes

        try:
            bar[i] = choice(size, p=p)
        except:
            print(p)
            print(temp_modifier)
            return None
            exit(1)

        j = bar[i]-prev[i]
        a[bar[i]] = 0
        used_jumps.append(j)

        if bar[i] % 12 != grundton % 12:
            used_notes[bar[i]::12] *= 0.3

    #print("jumps ", used_jumps)
    #print("notes ", used_notes)
    # bar[1] = choice(size,p=legal(prev[1],a))
    # a[bar[1]] = 0
    # bar[3] = choice(size,p=legal(prev[3],a))
    # a[bar[3]] = 0
    # bar[2] = choice(size,p=legal(prev[2],a))
    # a[bar[2]] = 0

    #print("bar   ", bar)
    return bar


def refine(bars):

    refines = []

    for bar, nextbar in zip(bars, bars[1:]+[[0, None, None, None, None]]):

        refbar = [0, 0, 0, 0, 0]
        for part in range(1, 5):

            refbar[part] = refined(bar[part], nextbar[part])

        refines.append(refbar)

    return refines


def refined(note, nextnote):

    if nextnote is not None:
        diff = nextnote - note

        if diff in [-3, -4, 3, 4]:
            if random(0.1):
                return [(note, 2), (note, 1), (towards(note, nextnote), 1)]
            elif random(0.3):
                return [(note, 3), (towards(note, nextnote), 1)]
        elif diff in [-5, 5]:
            if random(0.3):
                return [(note, 2), (towards(note, nextnote), 1), (towards(towards(note, nextnote), nextnote), 1)]

    return [(note, 4)]


def gen(temperature):
    start = [0, 24, 19, 16, 12]

    bars = [start]
    bar = start
    # order = list(np.random.choice((0,2,4,5,7,9,11),5))
    order = [0]
    total = 5+4
    for i in range(total):
        c = [0, 2, 4, 5, 7, 9, 11, -5]
        c.remove(order[i])
        if i == total-1:
            c.remove(-5)
        if order[i] == -5:
            a = 0
        else:
            a = np.random.choice(c)
        order.append(a)

    order.extend((7, 0))

    #print(order)

    # for a in [5,7,0]:
    i = 1
    while i < len(order):
        # for a in order:
        a = order[i]
        #print(a)
        nbar = next(bar, a, start=start, temperature=temperature)
        if nbar is not None:
            bar = nbar
            bars.append(bar)
            i += 1

    return bars


def export(bars, refined=True):

    s = Score()
    parts = []
    for i in range(1, 5):
        #        parts.append(s.add_child(Part(f"P{i}")))
        s.add_child(Part(f"P{i}"))

    parts = s.get_children()
    for bar in bars:
        for i, part in enumerate(parts):
            if refined:
                #                print(bar)
                for note, duration in bar[i+1]:
                    part.add_chord(Chord(int(midi_offset+note), duration))
            else:
                part.add_chord(Chord(int(midi_offset+bar[i+1]), 4))

    xml_path = Path(__file__).with_suffix('.xml')
    s.export_xml(xml_path)

    with open(xml_path, 'r') as f:
        lines = f.readlines()
        f.close()

    content = "".join(lines)
    seekstring = "<part-list>\n"
    plist_pos = content.find(seekstring)
    extra_parts = \
        """    <score-part id="P1">
      <part-name>P1</part-name>
    </score-part>
    <score-part id="P2">
      <part-name>P2</part-name>
    </score-part>
    <score-part id="P3">
      <part-name>P3</part-name>
    </score-part>\n"""

    content = content[:plist_pos+len(seekstring)] + \
        extra_parts + content[plist_pos+len(seekstring):]

    with open(xml_path, 'w') as f:
        f.write(content)
        f.close()

# print(legal(5,None))
# print(vector_to_note(legal(5,None)))
# print(vector_to_note(accord(5)))
#
# print(index_to_note(next([0,24,19,16,12],5)))


export(refine(gen(30)))
