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
#   Authored by:  Laura Murgatroyd (UKRI-STFC)


# An example of subclassing a CIL class!

from cil.recon import FDK


class FDK_subclass(FDK):
    ''' Example subclass of FDK class
    '''

    def __init__(self, *args, **kwargs):
        print("Initialising FDK subclass")
        super().__init__(*args, **kwargs)

    def run(self, out=None, verbose=1):
        print("Running with FDK subclass")
        return super().run(out, verbose)
