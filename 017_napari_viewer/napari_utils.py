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
# Tommi Heikkilä 2026, LUT

import napari
import cil
import numpy as np

class napari_viewer():
    """Simple wrapper to visualize data using Napari"""

    def __init__(self,
                datacontainer=None,
                name="data",
                rendering="mip",
                **kwargs):
        
        if datacontainer:
            ndim = datacontainer.ndim if datacontainer.ndim <= 3 else 3
            self.viewer = napari.Viewer(ndisplay=ndim)
            self.add_datacontainer(datacontainer, name=name, rendering=rendering, **kwargs)
        else:
            self.viewer = napari.Viewer()

    def add_datacontainer(self, datacontainer, name="data", **kwargs):
        """napari.add_image counterpart for CIL DataContainer"""
        dim_labels = datacontainer.dimension_labels
        units = self._get_data_units(datacontainer)
        scale = self._get_data_scale(datacontainer)
        self.viewer.add_image(datacontainer.as_array(),
                              name=name,
                              units=units,
                              scale=scale,
                              **kwargs)
        self._set_units_and_labels(dim_labels=dim_labels, units=units)
        return

    def _set_units_and_labels(self, dim_labels=None, units=None):
        """Helper function to set specific values and set them to be visible"""
        if dim_labels:
            self.viewer.dims.axis_labels = dim_labels
            self.viewer.axes.visible = True
        if units:
            self.viewer.scale_bar.unit = units
            self.viewer.scale_bar.visible=True
        return
    
    def add_separated_datacontainer(self, datacontainer, thresholds:list, style="value", name="data", 
                                    contrast_limits=None, **kwargs):
        """Split given data using threshold values and visualize them separately"""

        if style == "ratio": # Translate relative thresholds to value-based tresholds
            valThresholds = [datacontainer.max() * t for t in thresholds]
            return self.add_separated_datacontainer(datacontainer=datacontainer, thresholds=valThresholds, style="value",
                                                    name=name, contrast_limits=contrast_limits, **kwargs)
        
        min = datacontainer.min()
        max = datacontainer.max()

        if not contrast_limits:
            contrast_limits = [min, max]

        Ts = np.array(thresholds)
        if max > Ts.max():
            thresholds.append(max)
        if min < Ts.min():
            thresholds.append(min)
        thresholds.sort()

        dim_labels = datacontainer.dimension_labels
        units = self._get_data_units(datacontainer)
        scale = self._get_data_scale(datacontainer)
        self._set_units_and_labels(dim_labels=dim_labels, units=units)

        NoI = len(thresholds) - 1 # Number of intervals
        data = datacontainer.as_array()
        for n in range(NoI):
            subname = f"{name}{n}"
            a = thresholds[n]
            b = thresholds[n+1] if n+1 < NoI else thresholds[n+1] + 1 # +1 hack to include maximal values later
            self.viewer.add_image(np.where((a <= data) & (data < b), data, np.nan),
                                name=subname,
                                units=units,
                                scale=scale,
                                contrast_limits=contrast_limits, 
                                **kwargs)
        return


    def _get_data_scale(self, datacontainer):
        """Get pixel/voxel scale from CIL geometry. Angular pixels in sinograms have same size to
        retain square pixels."""
        g = datacontainer.geometry
        ndim = datacontainer.ndim
        if type(datacontainer) == cil.framework.ImageData:
            if ndim == 2:
                scale = [g.voxel_size_x, g.voxel_size_y]
            elif ndim == 3:
                scale = [g.voxel_size_x, g.voxel_size_y, g.voxel_size_z]
            else:
                scale = [1]*ndim
        elif type(datacontainer) == cil.framework.AcquisitionData:
            if ndim == 2:
                scale = [g.pixel_size_h, g.pixel_size_h]
            elif ndim == 3:
                scale = [g.pixel_size_v, g.pixel_size_h, g.pixel_size_h]
            else:
                scale = [1]*ndim
        else:
            scale = None
        return scale

    def _get_data_units(self, datacontainer):
        """Get pixel width (currently only one value for every dimension)"""
        g = datacontainer.geometry
        ndim = datacontainer.ndim
        if type(datacontainer) == cil.framework.ImageData:
            unit = "mm"
        elif type(datacontainer) == cil.framework.AcquisitionData:
            unit = g.config.units
            if unit == 'units distance':
                unit = ""
        else:
            unit = ""
        return unit