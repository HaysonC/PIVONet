#!/bin/bash

# Output folder
mkdir -p tmp

# Fields to export
fields=("rho" "u" "v" "p")

# Loop over all PyFR solution files
for f in double-mach-reflection-*.pyfrs; do
    # Extract timestep from filename
    step=$(echo "$f" | grep -o '[0-9]\+\.[0-9]\+')

    pyfr export volume double-mach-reflection.pyfrm  "$f" "tmp/double-mach-$step.vtu" 
done

echo "All PyFR timesteps exported to tmp/ folder!"
