#
#  Authored by:    Tommi Heikkilä (LUT)
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

from cil.optimisation.utilities import callbacks
from cil.optimisation.utilities.StepSizeMethods import StepSizeRule
from cil.optimisation.functions import L1Sparsity, ScaledFunction
import numpy as np

class ControlledSparsity(StepSizeRule):
    """Controlled (Wavelet Domains) Sparsity update rule.
    This only changes the regularization parameter for function `g`, NOT the step size! 
    The stopping criterion is given by `DesiredSparsity` callback.
    
    Reference
    ---------
        "Controlled wavelet domain sparsity for X-ray tomography" (2017), Purisha et al.  
        DOI:10.1088/1361-6501/aa9260
    """
    def __init__(self, desired_sparsity, step_weight=0.2, tol = 1e-8, print_stuff=0):
        """Initialize the Controlled (Wavelet Domains) Sparsity update rule."""
        self.tau = None
        self.step_size = None
        self.dif_sparsity = 1
        self.sparsity_vals = []
        self.reg_param_vals = []
        self.desired_sparsity = desired_sparsity # C_pr
        # These follow the notation from the article
        self.beta = step_weight
        self.kappa = tol
        # Print update on sparsity
        self.print_stuff = print_stuff
        self.W = None
        self.Wlen = None

    def get_step_size(self, algorithm):
        """Tune the regularization parameter (and hence the regularization parameter) using the sparsity of the current iterate"""
        g = algorithm.g
        if g.__class__ is not ScaledFunction:
            g = ScaledFunction(g, 1.0)
        self.tau = g.scalar
        regFun = g.function
        if not regFun.__class__ is L1Sparsity:
            raise TypeError(
                "The method does not necessarily make sense with regularizatio methods other than `L1Sparsity`!"
            )
        
        if self.W is None: # Get this operator if needed
            self.W = regFun.Q # Orthogonal operator, i.e. the wavelet transform
        if self.Wlen is None: # Get this value if needed
            self.Wlen = self._getWlen(self.W) # Normalization factor to get the ratio of "nonzero" coefficients
        if self.step_size is None: # Get this value if needed
            self.step_size = algorithm._calculate_default_step_size() # This actually never changes!

        Wx = self.W.direct(algorithm.get_output())
        sparsity = self._computeSparsity(Wx)
        self.sparsity_vals.append(sparsity)

        # Previous difference in sparsity
        dif_old = self.dif_sparsity
        # Current difference in sparsity
        dif = sparsity - self.desired_sparsity
        self.dif_sparsity = dif

        # If the sign changes, we have passed the desired sparsity -> smaller steps needed
        if np.sign(dif) != np.sign(dif_old):
            self.beta = self.beta*(1-np.minimum(0.99, np.abs(dif - dif_old)))

        # Update regularization parameter
        self.tau = np.maximum(0, self.tau + self.beta*dif)
        self.reg_param_vals.append(self.tau)

        if self.print_stuff > 0:
            print(f"Iteration : {algorithm.iterations[-1]} | Current sparsity: {sparsity:.3f} | Regularization param.: {self.tau:.2e} | Desired sparsity: {self.desired_sparsity:.3f} | Difference: {dif:.3f}") 

        # Here we update the regularization parameter `tau`
        g.scalar = self.tau
        algorithm.g = g
        # Return the same step size as before
        return self.step_size

    def _getWlen(self, W):
        '''Helper function to get the total number of wavelet coefficients (without the added zeros)'''
        ndim = W.domain_geometry().ndim
        shapes = W._shapes
        wLen = np.prod(shapes[0]) # Number of Approximation Coefficients
        for l in range(W.level):
            wLen += (2**ndim - 1)*np.prod(shapes[l+1][ndim*'d']) # Number of Detail Coefficients for level ´l´
        return wLen
    
    def _computeSparsity(self, Wx):
        '''Compute the number of "large" or "meaningful" coefficients given their magnitude'''
        return (Wx.abs().as_array() > self.kappa).sum() / self.Wlen
    

class DesiredSparsity(callbacks.Callback):
    """"Desired sparsity stopping rule, to halt the iteration when the desired sparsity
    level has been reached AND the relative change between consecutive iterates is small enough.
    This way the iterates have time to converge to the right regularization parameter value.
    
    Reference
    ---------
        "Controlled wavelet domain sparsity for X-ray tomography" (2017), Purisha et al.  
        DOI:10.1088/1361-6501/aa9260
    """
    def __init__(self, spar_tol=5e-3, rel_change_tol=1e-4, print_stuff=1, print_skip=5):
        self.sparsity_controller = None
        self.spar_tol = spar_tol # Sparsity tolerance
        self.rel_change_tol = rel_change_tol # Rel. change tolerance
        # Some values are still unknown
        self.x_old = -100.0 # Bogus value
        self.desired_sparsity = None
        self.current_sparsity = np.nan
        self.rel_change = np.inf
        # Print update on sparsity
        self.print_stuff = print_stuff
        self.print_skip = print_skip

    def __call__(self, algorithm):
        """Compute the latest sparsity difference and relative change"""
        iter = algorithm.iterations[-1] # Current iteration
        if self.sparsity_controller is None:
            self.sparsity_controller = algorithm.step_size_rule
        if self.desired_sparsity is None:
            self.desired_sparsity = self.sparsity_controller.desired_sparsity
        if iter < 0:
            return # Skip the first iterate
        self.current_sparsity = self.sparsity_controller.sparsity_vals[iter-1]

        # Difference in sparsity
        dif = self.current_sparsity - self.desired_sparsity
        self.dif_sparsity = dif

        self.rel_change = (algorithm.get_output() - self.x_old).norm()
        self.x_old = algorithm.get_output()

        # Print current status
        if iter % self.print_skip == 0:
            if self.print_stuff > 0:
                tau = algorithm.step_size_rule.tau
                print(f"Iteration : {iter} | Current sparsity: {self.current_sparsity:.3f} | Regularization param.: {tau:.2e} | Desired sparsity: {self.desired_sparsity:.3f} | Difference: {dif:.3f}", end='')
            if self.print_stuff > 1:
                beta = algorithm.step_size_rule.beta
                print(f" | Beta: {beta:.2e} | Relative change: {self.rel_change / self.x_old.norm():.2e}", end='')
            if self.print_stuff > 0:
                print("") # Line break
        
            
        # Check if current sparsity within tolerance
        if (np.abs(dif) < self.spar_tol) and self.rel_change < self.rel_change_tol*self.x_old.norm():
            print(f"Desired sparsity reached after {iter} iterations!")
            raise StopIteration