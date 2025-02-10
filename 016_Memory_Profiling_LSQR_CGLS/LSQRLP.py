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
#
# 

from itertools import count
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities.callbacks import Callback, LogfileCallback, _OldCallback, ProgressCallback
import numpy
import logging
from typing import List, Optional
import warnings
import math

import psutil
import os

log = logging.getLogger(__name__)


class LSQR_LP(Algorithm):

    r''' Least Squares QR (LSQR) algorithm
    
    The Least Squares QR (LSQR) algorithm is commonly used for solving large systems of linear equations, due to its fast convergence.

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

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    '''
    def __init__(self, initial=None, operator=None, data=None, **kwargs):
        '''initialisation of the algorithm
        '''
        self.process = psutil.Process(os.getpid())
        self.track_memory("Start of init")

        #We are deprecating tolerance 
        self.tolerance=kwargs.pop("tolerance", None)
        if self.tolerance is not None:
            warnings.warn( stacklevel=2, category=DeprecationWarning, message="Passing tolerance directly to CGLS is being deprecated. Instead we recommend using the callback functionality: https://tomographicimaging.github.io/CIL/nightly/optimisation/#callbacks and in particular the CGLSEarlyStopping callback replicated the old behaviour")
        else:
            self.tolerance = 0
        
        self.track_memory("1")
        super(LSQR_LP, self).__init__(**kwargs)
        self.track_memory("2")

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data) 
        self.track_memory("End of init", "self.set_up(initial=initial, operator=operator, data=data)")

    def track_memory(self, label="", line=" "):
        """Print memory usage with a custom label."""
        mem_info = self.process.memory_info().rss / (1024 * 1024)  # Memory in MB
        print(f"{label} | Memory Usage: {mem_info:.2f} MB | line: {line}")

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
        self.track_memory("Start of setup")
        log.info("%s setting up", self.__class__.__name__)
        self.x = initial.copy()
        self.track_memory("3", "self.x = initial.copy()")
        self.operator = operator
        self.track_memory("4", "self.operator = operator")

        self.u = data - self.operator.direct(self.x) # Forward proj?
        self.track_memory("5", "self.u = data - self.operator.direct(self.x)")
        self.beta = self.u.norm()
        self.track_memory("6", "self.beta = self.u.norm()")
        self.u = self.u/self.beta
        self.track_memory("7", "self.u = self.u/self.beta")
        
        self.v = self.operator.adjoint(self.u) # Backward proj?
        self.track_memory("8", "self.v = self.operator.adjoint(self.u)")
        self.alpha = self.v.norm()
        self.track_memory("9", "self.alpha = self.v.norm()")
        self.v = self.v/self.alpha
        self.track_memory("10", "self.v = self.v/self.alpha")

        self.c = 1
        self.s = 0
        self.track_memory("11", "self.c = 1, self.s = 0")

        self.phibar = self.beta
        self.track_memory("12", "self.phibar = self.beta")
        self.normr = self.beta
        self.track_memory("13", "self.normr = self.beta")
        
        self.d = operator.domain_geometry().allocate(0)
        self.track_memory("14", "self.d = operator.domain_geometry().allocate(0)")

        self.configured = True
        log.info("%s configured", self.__class__.__name__)
        self.track_memory("End of setup", 'log.info("%s configured", self.__class__.__name__)')

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
        self.track_memory("Start of run")
        if 'print_interval' in kwargs:
            warnings.warn("use `TextProgressCallback(miniters)` instead of `run(print_interval)`",
                 DeprecationWarning, stacklevel=2)
        if callbacks is None:
            callbacks = [ProgressCallback(verbose=verbose)]
        self.track_memory("15", "callbacks = [ProgressCallback(verbose=verbose)]")
        # transform old-style callbacks into new
        callback = kwargs.get('callback', None)
        self.track_memory("16", "callback = kwargs.get('callback', None)")

        if callback is not None:
            callbacks.append(_OldCallback(callback, verbose=verbose))
        if hasattr(self, '__log_file'):
            callbacks.append(LogfileCallback(self.__log_file, verbose=verbose))
        self.track_memory("17", "callbacks.append(_OldCallback(callback, verbose=verbose))")

        if self.should_stop():
            print("Stop criterion has been reached.")
        if iterations is None:
            warnings.warn("`run()` missing `iterations`", DeprecationWarning, stacklevel=2)
            iterations = self.max_iteration
        
        self.track_memory("18", "iterations = self.max_iteration")

        if self.iteration == -1 and self.update_objective_interval>0:
            iterations+=1
        self.track_memory("19", "iterations+=1")

        # call `__next__` upto `iterations` times or until `StopIteration` is raised
        self.max_iteration = self.iteration + iterations
        self.track_memory("20", "self.max_iteration = self.iteration + iterations")

        iters = (count(self.iteration) if numpy.isposinf(self.max_iteration)
                 else range(self.iteration, self.max_iteration))
        self.track_memory("21", "iters = (count(self.iteration) if numpy.isposinf(self.max_iteration) else range(self.iteration, self.max_iteration))")
        
        for _ in zip(iters, self):
            self.track_memory("22", "update(self)")
            try:
                for callback in callbacks:
                    callback(self)
                    self.track_memory("23", "callback(self)")
            except StopIteration:
                break
        self.track_memory("End of run")

    def update(self):
        self.track_memory("Start of update")
        '''single iteration'''
        # update u
        self.u = self.operator.direct(self.v) - self.alpha * self.u
        self.track_memory("24", "self.u = self.operator.direct(self.v) - self.alpha * self.u")
        self.beta = self.u.norm()
        self.track_memory("25", "self.beta = self.u.norm()")
        self.u = self.u/self.beta
        self.track_memory("26", "self.u = self.u/self.beta")

        # update scalars
        theta = -self.s * self.alpha
        self.track_memory("27", "theta = -self.s * self.alpha")
        rhobar = self.c * self.alpha
        self.track_memory("28", "rhobar = self.c * self.alpha")
        rho = math.sqrt((rhobar**2 + self.beta**2))
        self.track_memory("29", "rho = math.sqrt((rhobar**2 + self.beta**2))")

        self.c = rhobar/rho
        self.track_memory("30", "self.c = rhobar/rho")
        self.s = (-1*self.beta)/rho
        self.track_memory("31", "self.s = (-1*self.beta)/rho")
        phi = self.c*self.phibar
        self.track_memory("32", "phi = self.c*self.phibar")
        self.phibar = self.s*self.phibar
        self.track_memory("33", "self.phibar = self.s*self.phibar")

        # update d
        self.d.sapyb(-theta, self.v, 1, out=self.d)
        self.track_memory("34", "self.d.sapyb(-theta, self.v, 1, out=self.d)")
        self.d /= rho
        self.track_memory("35", "self.d /= rho")

        #update image x
        self.x.sapyb(1, self.d, phi, out=self.x)
        self.track_memory("36", "self.x.sapyb(1, self.d, phi, out=self.x)")

        # estimate residual norm
        self.normr = abs(self.s) * self.normr
        self.track_memory("37", "self.normr = abs(self.s) * self.normr")

        # update v
        self.v = self.operator.adjoint(self.u) - self.beta * self.v
        self.track_memory("38", "self.v = self.operator.adjoint(self.u) - self.beta * self.v")
        self.alpha = self.v.norm()
        self.track_memory("39", "self.alpha = self.v.norm()")
        self.v = self.v/self.alpha
        self.track_memory("End of update", "self.v = self.v/self.alpha")


    def update_objective(self):
        if self.normr is numpy.nan:
            raise StopIteration()
        self.loss.append(self.normr)

    def should_stop(self): # TODO: Deprecated, remove when CGLS tolerance is removed
        return self.flag() or super().should_stop()

    def flag(self): # TODO: Deprecated, remove when CGLS tolerance is removed
        flag = False

        if flag:
            self.update_objective()
            print('Tolerance is reached: {}'.format(self.tolerance))

        return flag