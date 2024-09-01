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
#   Authored by: Tommi Heikkilä (University of Helsinki, Finland),  Edooardo Pasca (UKRI-STFC),  



import numpy as np
import pywt # PyWavelets module

from cil.optimisation.operators import LinearOperator
from cil.optimisation.functions import Function  
from cil.framework import VectorData

###############################################################################
###############################################################################
########################## Discrete Wavelet Transform #########################
###############################################################################
###############################################################################              

class VectorGeometry:
    """
    Filler class for utilizing domain and range geometries with VectorData.
    VectorGeometry implements only the necessary attributes and methods such as
    `shape`, `voxel_num_x`, `allocate()` and `copy()`
    """
    def __init__(self, N):
        self.shape = (N,)
        self.voxel_num_x = N
    
    def allocate(self):
        return VectorData(np.empty(self.shape))
    
    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj


class WaveletOperator(LinearOperator):
    
    r'''                  
        Computes forward or inverse (adjoint) discrete wavelet transform of the input
        
        :param domain_geometry: Domain geometry for the WaveletOperator

        [OPTIONAL PARAMETERS]
        :param range_geometry: Output geometry for the WaveletOperator. 
            Default = domain_geometry with the right coefficient array size deduced from pywt
        :param level: integer for decomposition level.
            Default = log_2(min(shape(axes))), i.e. the maximum number of accurate downsamplings possible
        :type wname: string label for wavelet used.
            Default = "haar"
        :type axes: range of ints to define the dimensions to decompose along. Note that channel is the first dimension:
            For example, spatial DWT is given by axes=range(1,3) and channelwise DWT is axes=range(1)
            Default = None, meaning all dimensions are transformed. Same as axes=range(ndim)
        :param moments: integer for number of vanishing moments.
            Default = Known for Daubechies, None for others
        
     '''     
       
    def __init__(self, domain_geometry, 
                       range_geometry=None, 
                       level = None,
                       wname = "haar",
                       axes = None):
        
        if isinstance(domain_geometry, int):
            # Special case if domain_geometry is just the length of a 1D vector
            N = domain_geometry
            domain_geometry = VectorGeometry(N)

        
        if level is None:
            level = pywt.dwtn_max_level(domain_geometry.shape, wavelet=wname, axes=axes)
        self.level = int(level)

        self._shapes = pywt.wavedecn_shapes(domain_geometry.shape, wavelet=wname, level=level, axes=axes)
        self.wname = wname
        self.axes = axes
        self._slices = self._shape2slice()
        self.moments = pywt.Wavelet(wname).vanishing_moments_psi
        
        if range_geometry is None:
            range_geometry = domain_geometry.copy()
            shapes = self._shapes
            range_shape = np.array(domain_geometry.shape)
            if axes is None:
                axes = range(len(domain_geometry.shape))
            d = 'd'*len(axes) # Name of the diagonal element in unknown dimensional DWT
            for k in axes:
                range_shape[k] = shapes[0][k]
                for l in range(level):
                    range_shape[k] += shapes[l+1][d][k]

            # Update new size
            if hasattr(range_geometry, 'channels'):
                if range_geometry.channels > 1:
                    range_geometry.channels = range_shape[0]
                    range_shape = range_shape[1:]

            
            if len(range_shape) == 3:
                range_geometry.voxel_num_x = range_shape[2]
                range_geometry.voxel_num_y = range_shape[1]
                range_geometry.voxel_num_z = range_shape[0]
            elif len(range_shape) == 2:
                range_geometry.voxel_num_x = range_shape[1]
                range_geometry.voxel_num_y = range_shape[0]
            elif len(range_shape) == 1:
                range_geometry.voxel_num_x = range_shape[0]
                range_geometry.shape = (range_shape[0],)
            else:
                AttributeError(f"Dimension of range_geometry can be at most 3. Now it is {len(range_shape)}!")
                    
        super().__init__(domain_geometry=domain_geometry,range_geometry=range_geometry)
        

    def _shape2slice(self):
        """Helper function for turning shape of coefficients to slices"""
        shapes = self._shapes
        coeff_tmp = []
        coeff_tmp.append(np.empty(shapes[0]))

        for cd in shapes[1:]:
            subbs = dict((k, np.empty(v)) for k, v in cd.items())
            coeff_tmp.append(subbs)

        _, slices = pywt.coeffs_to_array(coeff_tmp, padding=0, axes=self.axes)
        return slices
    
    def _apply_weight(self, coeffs, weight):
        """
        Apply weight function to coefficients at different scales j.
        Note that the scaling coefficients are treated the same as the coarsest scale detail coefficients
        """

        # Scaling coefficients
        coeffs[0] = weight(0)*coeffs[0]
        # Detail coefficients
        for j,Cj in enumerate(coeffs[1:]):
            # All "directions" are treated the same per scale
            Cweighted = {k:weight(j)*c for (k,c) in Cj.items()}
            coeffs[j+1] = Cweighted

        
    def direct(self, x, out = None, weight = None, s = None):

        # Forward operator -- decomposition -- analysis
        
        x_arr = x.as_array()
        
        coeffs = pywt.wavedecn(x_arr, wavelet=self.wname, level=self.level, axes=self.axes)

        # Deduce weight from parameter s
        if (weight is None) and (s is not None):
            self._apply_weight(coeffs, weight = lambda j: 2**(j*s))
        # Apply given weight function
        elif weight is not None:
            self._apply_weight(coeffs, weight)
        # else: apply no weight
        # Note: weight takes priority over s

        Wx, _ = pywt.coeffs_to_array(coeffs, axes=self.axes)

        if out is None:
            ret = self.range_geometry().allocate()
            ret.fill(Wx)
            return ret
        else:
            out.fill(Wx) 
    
    def adjoint(self, Wx, out = None, weight = None, s = None):
        
        # Adjoint operator -- reconstruction -- synthesis
                      
        Wx_arr = Wx.as_array()
        coeffs = pywt.array_to_coeffs(Wx_arr, self._slices)

        # Deduce weight from parameter s
        if (weight is None) and (s is not None):
            self._apply_weight(coeffs, weight = lambda j: 2**(j*s))
        # Apply given weight function
        elif weight is not None:
            self._apply_weight(coeffs, weight)
        # else: apply no weight
        # Note: weight takes priority over s

        x = pywt.waverecn(coeffs, wavelet=self.wname, axes=self.axes)

        if out is None:
            ret = self.domain_geometry().allocate()
            ret.fill(x)
            return ret
        else:
            out.fill(x)
        
    def calculate_norm(self):
        orthWavelets = pywt.wavelist(family=None, kind="discrete")
        if self.wname in orthWavelets:
            norm = 1.0
        else:
            AttributeError(f"Unkown wavelet: {self.wname}! Norm not known.")
        return norm
    
 
def soft_shrinkage(x, tau, out=None):
    
    r"""Returns the value of the soft-shrinkage operator at x.
    """

    should_return = False
    if out is None:
        out = x.abs()
        should_return = True
    else:
        x.abs(out = out)
    out -= tau
    out.maximum(0, out = out)
    out *= x.sign()   

    if should_return:
        return out     


###############################################################################
###############################################################################
####################### L1-norm of Wavelet Coefficients #######################
###############################################################################
###############################################################################    
    
class WaveletNorm(Function):
    
    r"""WaveletNorm function
            
        Consider the following case:           
            a) .. math:: F(x) = ||Wx||_{1}
                                
    """   
           
    def __init__(self, W, weight = None, s = None):
        '''creator

        Cases considered :            
        a) :math:`f(x) = ||Wx||_{\ell^1}`
        b) :math:`f(x) = ||Wx||_{\ell^1(w)}` (weighted norm)

        :param W: Wavelet transform
        :type W: :code:`WaveletOperator`

        [OPTIONAL PARAMETERS]
        :param weight: function of scale j
        :param s: Besov norm smoothness parameter. This automatically implies weight:
            w(j) = 2^{js}
            NOTICE: if both `weight` and `s` are give, `weight` takes priority and `s` is discarded!
        '''
        super(WaveletNorm, self).__init__()
        self.W = W

        if (weight is None) and (s is None):
            # No weights
            def weight(j):
                return 1.0
        elif weight is None:
            # Define weight based on the Besov norm definition
            def weight(j):
                return 2**(j*s)
        self.weight = weight
        
    def __call__(self, x):
        
        r"""Returns the value of the WaveletNorm function at x.
        
        Consider the following case:           
            a) .. math:: f(x) = ||Wx||_{\ell^1}     
            b) .. math:: f(x) = ||Wx||_{\ell^1(w)} (weighted norm)
        """
        if self.weight is None:
            y = self.W.direct(x)
        else:
            y = self.W.direct(x, weight=self.weight)

        return y.abs().sum()  
          
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the WaveletNorm function at x.
        Here, we need to use the convex conjugate of WaveletNorm, which is the Indicator of the unit 
        :math:`\ell^{\infty}` norm on the Wavelet domain. (Since W is a basis of L^2).
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*})    
        
    
        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x^{*}\|_{\infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
    
        """        
        if self.weight is not None:
            NotImplementedError(f"Weighted norm convex conjugate not yet implemented!")
        
        tmp = self.W.direct(x).abs().max() - 1
        if tmp<=1e-5:            
            return 0.
        return np.inf

                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the WaveletNorm function at x.
        
        
        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = W^*\mathrm{ShinkOperator}(Wx)
    
        where,
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}
                            
        """  
        if self.weight is not None:
            NotImplementedError(f"Weighted norm proximal operator not yet implemented!")

        y = soft_shrinkage(self.W.direct(x), tau)
        if out is None:                                                
            return self.W.adjoint(y)
        else: 
            self.W.adjoint(y, out=out)
            return out