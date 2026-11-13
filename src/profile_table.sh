#!/bin/bash
# profile_table.sh -- the two profiler tables of a profile file, cut down to three columns
# so that they fit on a projector: total time in milliseconds, number of calls, name.
#
# First make the profile file:
#     nsys profile --stats=true --force-overwrite true -o blur_v1 ./04_blur_v1 ../photos/earth.jpg 100
# Then:
#     bash profile_table.sh blur_v1.nsys-rep
#
# It asks nsys for the same two reports as the long "nsys stats" command (the kernels, and the
# copies), with tabs between the columns, and keeps column 2 (total time, in nanoseconds),
# column 3 (number of calls) and column 9 (name).
file=${1:-blur_v1.nsys-rep}
if [ ! -f "$file" ]; then echo "No file '$file'. Run the nsys profile line first (see the top of this script)."; exit 1; fi
nsys stats --force-export=true --format tsv \
     --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum "$file" 2>/dev/null |
awk -F'\t' 'NF == 9 && $2 ~ /^Total/ { printf "\n%10s %7s   %s\n", "Total ms", "Calls", $9; next }
            NF == 9                  { printf "%10.1f %7d   %s\n", $2 / 1e6, $3, $9 }'
