#!/bin/bash
# Creates the job lists for the HTCondor DAG study.dag:
#   parameters_solve.txt: one line per (example, convectiontype, mu, stab1, stab2, level)
#   parameters_plot.txt:  one line per (example, mu, stab1, stab2), plots all convectiontypes
# Each line is "<request_memory> <arguments of scripts/run.jl>".
# Usage: ./create_parameters.sh && condor_submit_dag study.dag

rm -f parameters_solve.txt parameters_plot.txt
rm -rf log out err
mkdir log out err

examples=(p7vortex_powerlaw rigidbodyrotation)
convectiontypes=(NoConvection StandardConvection OseenConvection RotationForm KarperConvection NewConvection)
mu=(1 0.01)
stab1=("(0.9,0)")
stab2=("(1.5,0)")
nrefs_max=9

## memory per refinement level (each level has about 4 times the unknowns of the previous one),
## optional second argument: convectiontype (KarperConvection needs far more memory,
## it went over 16GB on level 8 and over 48GB on level 9)
memory() {
    if [ "$2" = "KarperConvection" ] && [ "$1" -ge 8 ]; then
        if [ "$1" -eq 8 ]; then echo 64GB; else echo 256GB; fi
    elif [ "$1" -le 6 ]; then echo 4GB
    elif [ "$1" -eq 7 ]; then echo 8GB
    elif [ "$1" -eq 8 ]; then echo 16GB
    else                      echo 48GB
    fi
}

## convection types joined by "+" for the plot jobs
all_convectiontypes=$(IFS=+; echo "${convectiontypes[*]}")

for ex in "${examples[@]}"; do
    for m in "${mu[@]}"; do
        for s1 in "${stab1[@]}"; do
            for s2 in "${stab2[@]}"; do
                for ct in "${convectiontypes[@]}"; do
                    for lvl in $(seq 1 $nrefs_max); do
                        echo "$(memory $lvl $ct) mode=solve example=$ex convectiontype=$ct mu=$m stab1=$s1 stab2=$s2 nrefs=$lvl" >> parameters_solve.txt
                    done
                done
                echo "$(memory $nrefs_max) mode=plot example=$ex convectiontypes=$all_convectiontypes mu=$m stab1=$s1 stab2=$s2 nrefs=1:$nrefs_max" >> parameters_plot.txt
            done
        done
    done
done

echo "Created parameters_solve.txt ($(wc -l < parameters_solve.txt) jobs) and parameters_plot.txt ($(wc -l < parameters_plot.txt) jobs)."
