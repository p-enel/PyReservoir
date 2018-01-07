#-*-coding: utf-8-*-
'''
Created on 10 déc. 2012

@author: pierre
'''
import matplotlib.pyplot as pl

def add_vertical_lines(positions, ax=None, **kwargs):
    '''Plot vertical lines that are fixed to an x value. Any arguments for line
    plotting can be used after the positions of the vertical lines
    '''
    if ax is None:
        ax = pl
    lines = []
    for pos in positions:
        lines.append(ax.axvline(x=pos, ymin=0, ymax=1, **kwargs))
    return lines

def add_vertical_shaded_areas(positions,
                              ax=None,
                              alpha=0.3,
                              facecolor='k',
                              **kwargs):
    if ax is None:
        ax = pl
    lines = []
    for start, end in positions:
        lines.append(ax.axvspan(start,
                                end,
                                facecolor=facecolor,
                                alpha=alpha,
                                **kwargs))
    
    return lines