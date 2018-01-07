'''
Created on Sep 29, 2014

@author: pier
'''

from __future__ import division
from ReservoirNSLNeurons import ReservoirNSL, np

class ReservoirNSLDelayedFB(ReservoirNSL):
    def test(self):
        '''Test the network once trained with a delayed feedback method
        
        Delayed feedback method means that the network will first make a choice
        in an open loop (no connections between the readout and the reservoir)
        and the choice will be fed back to the reservoir as a steady input
        after the choice period.
        
        Arguments:
        TO BE FILLED
        
        Returns:
        TO BE FILLED
        
        '''
        