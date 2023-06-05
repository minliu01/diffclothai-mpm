#!/bin/bash

start=$(date +%s)

python3 setup.py develop --user

end=$(date +%s)
runtime=$((end-start))

echo "Total runtime: $runtime seconds"
