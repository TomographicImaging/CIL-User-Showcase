#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from itertools import count
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities.callbacks import Callback, LogfileCallback, _OldCallback, ProgressCallback
import numpy
import logging
from typing import List, Optional
import warnings
import math

from memory_profiler import profile

log = logging.getLogger(__name__)


class CGLS_MP(Algorithm):

    r'''Conjugate Gradient Least Squares (CGLS) algorithm
    
    The Conjugate Gradient Least Squares (CGLS) algorithm is commonly used for solving large systems of linear equations, due to its fast convergence.

    Problem:

    .. math::

      \min_x || A x - b ||^2_2
      
      
    Parameters
    ------------
    operator : Operator
        Linear operator for the inverse problem
    initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
        Initial guess 
    data : DataContainer in the range of the operator 
        Acquired data to reconstruct
        
    Note
    -----
    Passing tolerance directly to CGLS is being deprecated. Instead we recommend using the callback functionality: https://tomographicimaging.github.io/CIL/nightly/optimisation/#callbacks and in particular the CGLSEarlyStopping callback replicated the old behaviour.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/cgls/
    '''
    @profile(precision=2)
    def __init__(self, initial=None, operator=None, data=None, **kwargs):
        '''initialisation of the algorithm
        '''
        #We are deprecating tolerance 
        self.tolerance=kwargs.pop("tolerance", None)
        if self.tolerance is not None:
            warnings.warn( stacklevel=2, category=DeprecationWarning, message="Passing tolerance directly to CGLS is being deprecated. Instead we recommend using the callback functionality: https://tomographicimaging.github.io/CIL/nightly/optimisation/#callbacks and in particular the CGLSEarlyStopping callback replicated the old behaviour")
        else:
            self.tolerance = 0
        
        super(CGLS_MP, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data) 

    @profile(precision=2)
    def set_up(self, initial, operator, data):
        r'''Initialisation of the algorithm
        Parameters
        ------------
        operator : Operator
            Linear operator for the inverse problem
        initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
            Initial guess 
        data : DataContainer in the range of the operator 
            Acquired data to reconstruct

        '''
        log.info("%s setting up", self.__class__.__name__)
        self.x = initial.copy()
        self.operator = operator

        self.r = data - self.operator.direct(self.x)
        self.s = self.operator.adjoint(self.r)

        self.p = self.s.copy()
        self.q = self.operator.range_geometry().allocate()

        self.norms0 = self.s.norm()
        self.norms = self.s.norm()

        self.gamma = self.norms0**2
        self.normx = self.x.norm()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @profile(precision=2)
    def run(self, iterations=None, callbacks: Optional[List[Callback]]=None, verbose=1, **kwargs):
        r"""run upto :code:`iterations` with callbacks/logging.
        
        For a demonstration of callbacks see https://github.com/TomographicImaging/CIL-Demos/blob/main/misc/callback_demonstration.ipynb

        Parameters
        -----------
        iterations: int, default is None
            Number of iterations to run. If not set the algorithm will run until :code:`should_stop()` is reached
        callbacks: list of callables, default is Defaults to :code:`[ProgressCallback(verbose)]`
            List of callables which are passed the current Algorithm object each iteration. Defaults to :code:`[ProgressCallback(verbose)]`.
        verbose: 0=quiet, 1=info, 2=debug
            Passed to the default callback to determine the verbosity of the printed output. 
        """

        if 'print_interval' in kwargs:
            warnings.warn("use `TextProgressCallback(miniters)` instead of `run(print_interval)`",
                 DeprecationWarning, stacklevel=2)
        if callbacks is None:
            callbacks = [ProgressCallback(verbose=verbose)]
            
        # transform old-style callbacks into new
        callback = kwargs.get('callback', None)

        if callback is not None:
            callbacks.append(_OldCallback(callback, verbose=verbose))
        if hasattr(self, '__log_file'):
            callbacks.append(LogfileCallback(self.__log_file, verbose=verbose))

        if self.should_stop():
            print("Stop criterion has been reached.")
        if iterations is None:
            warnings.warn("`run()` missing `iterations`", DeprecationWarning, stacklevel=2)
            iterations = self.max_iteration

        if self.iteration == -1 and self.update_objective_interval>0:
            iterations+=1

        # call `__next__` upto `iterations` times or until `StopIteration` is raised
        self.max_iteration = self.iteration + iterations

        iters = (count(self.iteration) if numpy.isposinf(self.max_iteration)
                 else range(self.iteration, self.max_iteration))
        
        for _ in zip(iters, self):
            try:
                for callback in callbacks:
                    callback(self)
            except StopIteration:
                break

    # @profile(precision=2)
    def update(self):
        '''single iteration'''

        self.operator.direct(self.p, out=self.q)
        delta = self.q.squared_norm()
        alpha = self.gamma/delta

        self.x.sapyb(1, self.p, alpha, out=self.x)
        #self.x += alpha * self.p
        self.r.sapyb(1, self.q, -alpha, out=self.r)
        #self.r -= alpha * self.q

        self.operator.adjoint(self.r, out=self.s)

        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        #self.p = self.s + self.beta * self.p
        self.p.sapyb(self.beta, self.s, 1, out=self.p)

        self.normx = self.x.norm()# TODO: Deprecated, remove when CGLS tolerance is removed


    def update_objective(self):
        a = self.r.squared_norm()
        if a is numpy.nan:
            raise StopIteration()
        self.loss.append(a)

    def should_stop(self): # TODO: Deprecated, remove when CGLS tolerance is removed
        return self.flag() or super().should_stop()

    def flag(self): # TODO: Deprecated, remove when CGLS tolerance is removed
        '''returns whether the tolerance has been reached'''
        flag  = (self.norms <= self.norms0 * self.tolerance) or (self.normx * self.tolerance >= 1)

        if flag:
            self.update_objective()
            print('Tolerance is reached: {}'.format(self.tolerance))

        return flag