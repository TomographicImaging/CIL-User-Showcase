# -*- coding: utf-8 -*-
#  Copyright 2024 Till Leissner
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
#   Authored by: Till Leissner (University of Southern Denmark)
#   Edited by: 
#
#   We acknowledge support from the ESS Lighthouse on Hard Materials in 3D, SOLID, funded by the Danish Agency for Science and Higher Education (grant No. 8144-00002B).


### General imports 
import numpy as np
import glob
import os

### Import all CIL components needed
from cil.framework import AcquisitionGeometry



def set_geometry(logfile,infofile=None):
    """
    Creates a CIL AcquisitionGeometry object for a cone-beam CT scan.

    Args:
        infofile (str): Path to the information file.
        logfile (str): Path to the log file.

    Returns:
        AcquisitionGeometry: A geometry object representing the scan setup.
    """

    # Extract scan parameters from the log file
    scanparams = get_scanparams(logfile)

    # Get angles in degrees and radians
    if infofile:
        theta_deg, theta_rad = get_angles_from_infofile(infofile)
    else: 
        theta_deg, theta_rad = get_angles_from_logfile(scanparams)


    # Calculate distances
    distance_source_origin = float(scanparams['Acquisition']['object_to_source_(mm)'])
    distance_origin_detector = float(scanparams['Acquisition']['camera_to_source_(mm)'])-distance_source_origin
    
    # Get detector properties
    detector_pixel_size = float(scanparams['System']['camera_pixel_size_(um)'])/1000
    detector_rows = int(scanparams['Acquisition']['number_of_rows'])
    detector_cols = int(scanparams['Acquisition']['number_of_columns'])

    # Create AcquisitionGeometry object
    ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-distance_source_origin,0],detector_position=[0,distance_origin_detector,0],
                                       detector_direction_x=[1,0,0],detector_direction_y=[0,0,1]
                                      )\
    .set_panel(num_pixels=[detector_cols,detector_rows], pixel_size = detector_pixel_size)\
    .set_angles(angles=theta_deg)

    return ag




def get_angles_from_logfile(scanparams):
    """
    Calculates angles in degrees from the information in the log file.

    Args:
        logfile (str): Path to the log file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - First array: Angles in degrees.
            - Second array: Angles converted to radians.
    """
    angles_deg = [ i*float(scanparams['Acquisition']['rotation_step_(deg)']) for i in range(int(scanparams['Acquisition']['number_of_files'])) ]
    return np.array(angles_deg), np.deg2rad(angles_deg)




def get_angles_from_infofile(infofile, startangle=0):
    """
    Extracts angles in degrees from an information file.

    Args:
        infofile (str): Path to the information file.
        startangle (float, optional): Starting angle in degrees (default is 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - First array: Angles in degrees.
            - Second array: Angles converted to radians.
    """
    
    # Initialize the list of angles with the start angle
    angles_deg = [startangle]
    
    # Read the information file
    with open(infofile, "r") as f:
        for line in f.readlines():
                line = line.rstrip('\n')
                if 'Drift compensation scan' in line:
                        break
                elif 'achieved angle' in line:
                        angles_deg.append(float(line.strip().rsplit(' ',1)[-1][:-1]))
    return np.array(angles_deg), np.deg2rad(angles_deg)



def get_scanparams(logfile):
    """
    Extracts scan parameters from a log file.

    Args:
        logfile (str): Path to the log file.

    Returns:
        dict: A dictionary containing scan parameters organized by sections.
    """
    
    # Initialize an empty dictionary to store scan parameters
    scanparams = {}

    # Read the log file
    with open(logfile, "r") as f:
        for line in f.readlines():
                line = line.rstrip('\n')
                if line.startswith('['):
                        section = line[1:][:-1]
                        scanparams[section] ={}
                else:
                        scanparams[section][str.split(line, '=')[0].replace(' ','_').lower()]=str.split(line, '=')[1]
        return scanparams
        

def get_filelist(datadir,dataset_prefix,num_digits=8, extension='tif'):
    """
    Retrieves a sorted list of file paths matching a specific pattern.

    Args:
        datadir (str): Directory path where the files are located.
        dataset_prefix (str): Prefix for the dataset filenames.
        num_digits (int, optional): Number of digits in the numeric part of the filenames (default is 8).
        extension (str): File extension (default is 'tif').

    Returns:
        List[str]: A sorted list of file paths matching the specified pattern.
    """
    return sorted(glob.glob(datadir+'/'+dataset_prefix+('[0-9]' * num_digits)+'.'+extension))



def rename_files(directory, old_prefix, new_prefix):
    #Usage: bc.rename_files('./samples/02-switch-large', '001', '001_')
    for filename in os.listdir(directory):
        if filename.startswith(old_prefix):
            new_name = filename.replace(old_prefix, new_prefix, 1)
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))


