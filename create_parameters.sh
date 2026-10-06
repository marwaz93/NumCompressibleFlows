#!/bin/bash

rm parameters.txt
rm -r log out err
mkdir log out err

gamma=(1 1.4 3)
mu=(1 0.1 )
c=(1 100)
M=(1.0)
tau_fac=(10.0)
stab1=("(0.9,0.01)")
nrefs=("1:4")
reconstruct=(true false)
velocitytype=(P7VortexVelocity)

for g in "${gamma[@]}"; do
    for m in "${mu[@]}"; do
        for cc in "${c[@]}"; do
            for mm in "${M[@]}"; do
                for tf in "${tau_fac[@]}"; do
                    for s in "${stab1[@]}"; do
                        for nr in "${nrefs[@]}"; do
                            for r in "${reconstruct[@]}"; do
                                for vt in "${velocitytype[@]}"; do
                                    echo "gamma=$g mu=$m c=$cc M=$mm tau_fac=$tf stab1=$s nrefs=$nr reconstruct=$r velocitytype=$vt" >> parameters.txt
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Created parameters.txt."