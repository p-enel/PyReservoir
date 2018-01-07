# -*- coding: utf-8 -*-
'''
Created on 15 déc. 2012

@author: pier
'''
from distutils.core import setup

setup(name='PyReservoir',
      description='Simple reservoir computing (ESN) tools',
      author='Pierre Enel',
      author_email='pierre.enel@inserm.fr',
      packages=['PyReservoir',
                'PyReservoir.learning',
                'PyReservoir.reservoir',
                'PyReservoir.visualization']
     )
