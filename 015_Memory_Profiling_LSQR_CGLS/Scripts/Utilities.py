# Imports
from cil.processors import Slicer
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import time
import re

# Functions to process the memory profiling outputs:
def mem_prof_process(output, norm=False):
    mem_prof = output.stdout

    # split text into lines
    blocks = re.split(r"Filename: .+", mem_prof)[1:]
    pattern = r"\s*(\d+)\s+([\d.]*){1}(\s*MiB\s*){1}(-?[\d.]+){1}(\s*MiB\s*){1}(\d+){1}(\s+.*)"
    repNum = None
   
    for idx, block in enumerate(blocks):
        # split the string into lines
        lines = block.split('\n')    
        memusages = []
            
        # iterate over each line/string
        for line in lines:          
            memUsgMatch = re.match(pattern, line)
            funcNmMatch = re.match(r"\s*(\d+)\s*(def (.+)\(.*)", line)
            
            if funcNmMatch:
                funcName = funcNmMatch.group(2)
                method = funcNmMatch.group(3)
                if method == "update":
                    rearr=True
                if repNum is not None:
                    method = f"{method} {repNum}"

            if memUsgMatch:
                memusages.append(float(memUsgMatch.group(2)))

            repMatch = re.match(r"(\w+) (\d+)", line)
            if repMatch:
                repNum = repMatch.group(2)

        blocks[idx] = [method, funcName.strip('\r'), memusages[0], memusages[-1]]
    
    rearrBlocks = list(reversed(blocks[:2]))
    currGroup = []
    for i, item in enumerate(blocks[2:]):
        method = item[0]
    
        repNumMatch = re.match(r'\w+ (\d+)', method)
        if repNumMatch:
            repNum = int(repNumMatch.group(1))
    
            if not currGroup:
                currGroup.append(item)
                
            elif repNum == int(re.search(r'\w+ (\d+)', currGroup[-1][0]).group(1)):
                currGroup.append(item)
            else:
                rearrBlocks.extend(list(reversed(currGroup)))
                currGroup = [item]
            if i == len(blocks[2:])-1:
                rearrBlocks.extend(list(reversed(currGroup)))       
        else:
            rearrBlocks.append(item)

    mp_df = DataFrame(rearrBlocks, columns=['Method', 'Method Call', 'Initial Memory Usage (MiB)', 'Final Memory Usage (MiB)'])
    if norm:
        mp_df = normalise_df(mp_df)
    mp_df['Total Memory Usage (MiB)'] = mp_df['Final Memory Usage (MiB)'] - mp_df['Initial Memory Usage (MiB)']

    return mp_df

def line_prof_process(output, norm=False):
    line_prof = output.stdout
    df = {'Method': ['init', 'setup'], 'Initial Memory Usage (MiB)': [], 'Final Memory Usage (MiB)': []}
    memory_usage = {'init': [], 'setup': [], 'run': []}
    countrun = 1

    for res in line_prof.splitlines():
        match = re.match(r"(\w+) of (\w+).+ ([0-9]+\.[0-9]+).+", res)
        if match:
            action, method, value = match.groups()
            value = float(value)

            if action == 'Start' and method =='run' and method not in df['Method']:
                df['Method'].append(f"{method} {countrun}")
                countrun += 1

            if method.startswith('run'):
                memory_usage['run'].append(value)
            elif method in memory_usage:
                memory_usage[method].append(value)

    # Populate initial and final memory usage
    df['Initial Memory Usage (MiB)'] = [memory_usage['init'][0], memory_usage['setup'][0]] + memory_usage['run'][::2]
    df['Final Memory Usage (MiB)'] = [memory_usage['init'][1], memory_usage['setup'][1]] + memory_usage['run'][1::2]

    lp_df = DataFrame(df)
    if norm:
        lp_df = normalise_df(lp_df)
    lp_df['Total Memory Usage (MiB)'] = lp_df['Final Memory Usage (MiB)'] - lp_df['Initial Memory Usage (MiB)']
    
    return lp_df

def normalise_df(df):
    # Access the start memory from the "init" method
    startMem = df["Initial Memory Usage (MiB)"].iat[0]
    # subtract startMem from all values in the df
    df[df.select_dtypes(include=['number']).columns] -= startMem # Done BEFORE computing Total Mem Usg

    return df 

def split_groups(line_prof_peak):
    pattern = r"(\w+) of (\w+).+ ([0-9]+\.[0-9]+).+"
    groups = {}
    stack = []
    methodrep = None
    
    for line in line_prof_peak.splitlines():
        repmatch = re.match(r"(^\w*) ([0-9]+$)", line)
        if repmatch: # Find repetitions of methods i.e. run/update
            methodrep = (repmatch.group(1), repmatch.group(2))   

        match = re.match(pattern, line) # Find line-by-line outputs
        if match:
            action, method, value = match.groups()
            value = float(value)
            if methodrep:
                method = f"{method} {methodrep[1]}"

            if action == "Start": # Stores new method encounter & adds to top of stack
                groups[method] = []
                stack.append(method)
                groups[method].append(line)
            elif action == "End": # Remove current method from top of stack
                groups[method].append(line)
                stack.pop(-1)

        elif stack: # Put all lines with the method on top of stack
            method = stack[-1]
            groups[method].append(line)
            if line.startswith("Memory Usage Log"): # Add background/peak memory lines to parent & child methods' group
                for parent_method in stack[:-1]: 
                    groups[parent_method].append(line)
            
    return groups

def line_peak_process(output, norm=False):
    line_prof_peak = output.stdout
    df = {'Method': [], 'Initial Memory Usage (MiB)': [], 'Peak Memory': [], 'Final Memory Usage (MiB)': [], 'Peak Line': []}
    groups = split_groups(line_prof_peak)
    pattern = r"(\w+) of (\w+).+ ([0-9]+\.[0-9]+).+"
    
    for method, block in groups.items():  # iterate over keys [init, start, run1, update1, run2, update2 ...]
        peakline = ""
        awaiting_line = True
        methodpeak = 0
        r = re.compile(pattern) # Get start and end memory usages for current method's block
        start, end = list(filter(r.match, block))
        startmem = float(re.match(pattern, start).group(3))
        endmem = float(re.match(pattern, end).group(3))

        df['Method'].append(method)
        df['Initial Memory Usage (MiB)'].append(startmem)
        df['Final Memory Usage (MiB)'].append(endmem)

        for line in block: # Find peak memory usage in current block 
            peakmatch = re.match(r".+ (\d+\.\d+)\s*MB.*", line)
            if peakmatch:
                currmem = float(peakmatch.group(1))
                if currmem > methodpeak:
                    methodpeak = currmem
                    awaiting_line = True

            codematch = re.match(r".+ \| Memory Usage: (\d+.\d+) MB \| line:(.+)+", line)
            if codematch: # Find the line-by-line output to attribute to the peak usage
                currmem = float(codematch.group(1))
                if currmem > methodpeak:
                    methodpeak = currmem
                    peakline = codematch.group(2).strip()
                    awaiting_line = False 
                elif awaiting_line:  
                    peakline = codematch.group(2).strip()
                    awaiting_line = False

        df['Peak Memory'].append(methodpeak)
        df['Peak Line'].append(peakline)

    df = DataFrame(df)
    if norm:
        df = normalise_df(df)
    df.insert(4, 'Total Memory Usage (MiB)', df['Final Memory Usage (MiB)']-df['Initial Memory Usage (MiB)'])
    df.insert(5, 'Peak Memory Increase (MiB)', df['Peak Memory']-df['Initial Memory Usage (MiB)'])
    
    return df

# Function to plot the memory profiling DataFrame(s):
def plot_mem(dfs, algNms, norm=False, linestyles=("-", "--", "--")):

    plt.figure(figsize=(10, 6))

    for i, (df, algNm) in enumerate(zip(dfs, algNms)):
        df = df.iloc[::2]

        # Flatten data for sequential plotting
        methods = []
        stages = []
        values = []

        for _, row in df.iterrows():
            methods.extend([row["Method"]] * 3)
            stages.extend(["Initial", "Peak", "Final"])
            values.extend([row["Initial Memory Usage (MiB)"], row["Peak Memory"], row["Final Memory Usage (MiB)"]])

        if norm == True:
            startMem = df["Initial Memory Usage (MiB)"].iat[0]
            values = np.asarray(values) - startMem
            
        plt.plot(range(len(values)), values, label=algNm, marker="o", linestyle=linestyles[i])

    # x-axis method labels
    xticksPos = [i for i in range(len(values))]
    xticksLabels = [f"{method} - {stage}" for method, stage in zip(methods, stages)]

    plt.xticks(xticksPos, xticksLabels, rotation=60, ha="right", fontsize=10)

    # titles & labels
    plt.title(f"Memory Usage by Method (Sequential)", fontsize=14)
    plt.xlabel("Methods and Memory Stages", fontsize=12)
    plt.ylabel("Memory (MB)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # highlight transitions between methods
    for i in range(0, len(values), 3):
        plt.axvline(i - 0.5, color="orange", linestyle="--", alpha=0.5)  # divider between methods

    plt.tight_layout()
    plt.legend()
    plt.show()


# Function for returning the time, residual & relative error of each iteration
def timed_iterations(dataset, A, sino_ground_truth, recon_ground_truth, padsize, algorithm, iterations, itsAtATime):
    padend= dataset.shape[1]-padsize
    roi = {'horizontal':(padsize, padend)}
    roi_xy = {'horizontal_y':(padsize, padend), 'horizontal_x':(padsize, padend)}

    sino_norm = Slicer(roi)(sino_ground_truth).norm() # crop the 'perfect' sinogram
    recon_norm = recon_ground_truth.norm() # 'perfect' reconstruction, already cropped

    # Setting up output arrays
    times = np.zeros(iterations)
    residuals = np.zeros(iterations)
    errors = np.zeros(iterations)
    time_tot = 0
    
    for ii in range(iterations): 
        start = time.time()
        algorithm.run(itsAtATime, verbose=False)
        end = time.time()

        solutncrop = Slicer(roi_xy)(algorithm.solution)
        directcrop = Slicer(roi)(A.direct(algorithm.solution))
        dataset_crop = Slicer(roi)(dataset)
        
        residual = directcrop - dataset_crop
        
        res_norm = residual.norm()
        rel_res = res_norm/sino_norm     
        residuals[ii] = rel_res

        # The solution needs to be cropped, recon_ground_truth is already cropped
        error = solutncrop - recon_ground_truth
        err_norm = error.norm()
        rel_err = err_norm/recon_norm
        errors[ii] = rel_err

        time_tot += end-start
        times[ii] = time_tot     
    # show2D([solutncrop, directcrop])
    return times, residuals, errors