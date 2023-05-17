# -*- coding: utf-8 -*-
#  Copyright 2023 Physikalisch-Technische Bundesanstalt (PTB)
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
#   Authored by:  Christoph Kolbitsch (PTB)

from cil.plugins.ccpi_regularisation.functions import  FGP_TV
from cil.framework import  ImageGeometry

import numpy as np

class BDCtoID():
    
    def __init__(self, bdc):
        
        self.bdc = bdc
        self.bdc_shape = bdc[0].shape
        
        # check that all datacontainers have the same shape
        all_shape = [i.shape for i in bdc.containers]
        num_containers = len(bdc.containers)
        if len(set(all_shape))>1:
            raise ValueError("Different shapes for containers")            
        if len(self.bdc_shape)==2:
            y, x = self.bdc_shape
            self.ig = ImageGeometry(voxel_num_y = y, voxel_num_x = x, channels=num_containers)
        elif len(self.bdc_shape)==3:            
            z,y,x = self.bdc_shape
            self.ig = ImageGeometry(voxel_num_z = z, voxel_num_y = y, voxel_num_x = x, 
                                    channels=num_containers)
        else:
            raise ValueError("not implemented")
        
    def IDarray(self, block_data_cont):
        tmp = []
        for i in range(block_data_cont.shape[0]):
            tmp.append(block_data_cont[i].as_array())
        tmp = np.stack(tmp, axis=0)
        self.id_cil = self.ig.allocate(dtype=block_data_cont[0].dtype)
        self.id_cil.fill(np.squeeze(tmp))
        return self.id_cil
    
    def BDC(self, image_data):
        if image_data.ndim<=2:
            raise ValueError("We cannot create a BlockDataContainer from a 2D array")
            
        num_slices = image_data.shape[0]
        splitted_arrays = np.squeeze(np.split(image_data.array, num_slices, axis=0))
        
        # reshape to make sure single dimensions are restored
        splitted_arrays = [np.reshape(arr, self.bdc_shape) for arr in splitted_arrays]
        
        # create new data container
        bdc = self.bdc.copy()
        for idx, container in enumerate(bdc.containers):
            container.fill(splitted_arrays[idx])
        return(bdc)
    
    
    
class FGP_TV_BDC():
    def __init__(self, bdc, alpha):
        self.bdctoid = BDCtoID(bdc) 
        self.fgp_tv = alpha*FGP_TV(nonnegativity=False)
        
    def __call__(self, bdc):
        x = self.bdctoid.IDarray(bdc)
        x_real = x.copy()
        x_real.fill(x.as_array().real)
        x_imag = x.copy()
        x_imag.fill(x.as_array().imag)
        return(np.sqrt(self.fgp_tv(x_real)**2 + self.fgp_tv(x_imag)**2))
    
    def proximal_numpy(self, xarr, tau):
        return(self.fgp_tv.proximal_numpy(xarr, tau))
        
    def proximal(self, bdc, tau, out=None):  
        x = self.bdctoid.IDarray(bdc)
        arr = x.as_array().copy() # .copy() added
        prox = x.copy()
        if np.iscomplexobj(arr):
            # do real and imag part indep
            in_arr = np.asarray(arr.real, dtype=np.float32, order='C')
            res, info = self.proximal_numpy(in_arr, tau)
            arr.real = res[:]
            in_arr = np.asarray(arr.imag, dtype=np.float32, order='C')
            res, info = self.proximal_numpy(in_arr, tau)
            arr.imag = res[:]
            self.info = info
            prox.fill(arr)
        else:
            arr = np.asarray(x.as_array(), dtype=np.float32, order='C')
            res, info = self.proximal_numpy(arr, tau)
            self.info = info
            prox.fill(res)
        
        if out is not None:
            out.fill(self.bdctoid.BDC(prox))
        else:
            out = self.bdctoid.BDC(prox)
            return out