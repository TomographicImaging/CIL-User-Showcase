# -*- coding: utf-8 -*-
#  Copyright 2023 United Kingdom Research and Innovation
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
#   Authored by:  Bill Lionheart (University of Manchester),
#   Edited by:  Margaret Duff (STFC - UKRI)



import numpy as np

    
def deriv2(n ,example=1):
    #DERIV2 Test problem: computation of the second derivative.
    
    # [A,b,x] = deriv2(n,example)
    
    # This is a mildly ill-posed problem.  It is a discretization of a
# first kind Fredholm integral equation whose kernel K is the
# Green's function for the second derivative:
#    K(s,t) = | s(t-1)  ,  s <  t .
#             | t(s-1)  ,  s >= t
# Both integration intervals are [0,1], and as right-hand side g
# and correspond solution f one can choose between the following:
#    example = 1 : g(s) = (s^3 - s)/6          ,  f(t) = t
#    example = 2 : g(s) = exp(s) + (1-e)s - 1  ,  f(t) = exp(t)
#    example = 3 : g(s) = | (4s^3 - 3s)/24               ,  s <  0.5
#                         | (-4s^3 + 12s^2 - 9s + 1)/24  ,  s >= 0.5
#                  f(t) = | t    ,  t <  0.5
#                         | 1-t  ,  t >= 0.5
    
    # References.  The first two examples are from L. M. Delves & J. L.
# Mohamed, "Computational Methods for Integral Equations", Cambridge
# University Press, 1985; p. 310.  The third example is from A. K.
# Louis and P. Maass, "A mollifier method for linear operator equations
# of the first kind", Inverse Problems 6 (1990), 427-440.
    
    # Discretized by the Galerkin method with orthonormal box functions.
    
    # Per Christian Hansen, IMM, 04/21/97.
    
    # Initialization.
    #if (len(varargin) == 1):
    #example = 1
    
    h = 1 / n
    sqh = np.sqrt(h)
    h32 = h * sqh
    h2 = h ** 2
    sqhi = 1 / sqh
    t = 2 / 3
    A = np.zeros((n,n))
    # Compute the matrix A.
    for i in np.arange(0,n).reshape(-1):
        A[i,i] = h2 * ((i ** 2 - i + 0.25) * h - (i - t))
        for j in np.arange(0,i).reshape(-1):
            A[i,j] = h2 * (j - 0.5) * ((i - 0.5) * h - 1)
    
    A = A + np.transpose(np.tril(A,-1))
    # Compute the right-hand side vector b.
#  if ( 1):
#      b = np.zeros((n,1))
#      if (example == 1):
#          for i in np.arange(1,n).reshape(-1):
#              b[i] = h32 * (i - 0.5) * ((i ** 2 + (i - 1) ** 2) * h2 / 2 - 1) / 6
#      else:
#          if (example == 2):
#              ee = 1 - np.exp(1)
#              for i in np.arange(1,n).reshape(-1):
#                  b[i] = sqhi * (np.exp(i * h) - np.exp((i - 1) * h) + ee * (i - 0.5) * h2 - h)
#          else:
#              if (example == 3):
#                  if (rem(n,2) != 0):
#                      raise Exception('Order n must be even')
#                  else:
#                      for i in np.arange(1,n / 2+1).reshape(-1):
#                          s12 = (i * h) ** 2
#                          s22 = ((i - 1) * h) ** 2
#                          b[i] = sqhi * (s12 + s22 - 1.5) * (s12 - s22) / 24
#                      for i in np.arange(n / 2 + 1,n).reshape(-1):
#                          s1 = i * h
#                          s12 = s1 ** 2
#                          s2 = (i - 1) * h
#                          s22 = s2 ** 2
#                          b[i] = sqhi * (- (s12 + s22) * (s12 - s22) + 4 * (s1 ** 3 - s2 ** 3) - 4.5 * (s12 - s22) + h) / 24
#              else:
#                  raise Exception('Illegal value of example')
    
    # Compute the solution vector x.

    x = np.zeros((n,1))
    if (example == 1):
            for i in np.arange(1,n).reshape(-1):
                x[i] = h * (i )
    else:
            if (example == 2):
                for i in np.arange(1,n).reshape(-1):
                    x[i] = sqhi * (np.exp(i * h) - np.exp((i - 1) * h))
            else:
                for i in np.arange(1,int(np.floor(n/ 2))).reshape(-1):
                    x[i] = sqhi * ((i * h) ** 2 - ((i - 1) * h) ** 2) / 2
                for i in np.arange(int(np.floor(n / 2)) ,n).reshape(-1):
                    x[i] = sqhi * (h - ((i * h) ** 2 - ((i - 1) * h) ** 2) / 2)
    b= A@x; #  So solution is correct up to fp error
    
    return A,b.flatten(),x.flatten()   # flatten so vectors rather than matrices
