#!/usr/bin/env bash

# If single example is launched, launch that
if [[ $# -eq 3 ]]; then
    python3 analyzer.py ../mnist_nets/$1.txt ../mnist_images/img$2.txt $3
    exit $?
fi

# If test not passed as argument
if [ "$1" != "test" ]; then
    echo "Please refer to the README for correct launch instructions."
    exit 0
fi

# Declare variables
declare -a nets=("mnist_relu_3_10" "mnist_relu_3_20" "mnist_relu_3_50" "mnist_relu_4_1024"
                 "mnist_relu_6_20" "mnist_relu_6_50" "mnist_relu_6_100" "mnist_relu_6_200"
                 "mnist_relu_9_100" "mnist_relu_9_200")
declare -i total=0
declare -i passed=0


# Loop over nets
for net in "${nets[@]}"; do
    echo "Processing net '${net}'"
    # Loop over images
    for img_num in {0..99}; do
        # Loop over epsilon values
        for eps in 0.005 0.01 0.02 0.04 0.08 0.1; do
            ((++total))
            # Run the analyzer
            python3 analyzer.py ../mnist_nets/$net.txt ../mnist_images/img$img_num.txt $eps > /dev/null

            # Check return code and increase if passed
            if [[ $? -eq 0 ]]; then
                ((++passed))
            fi
        done
    done
done

# Compute passed percentage
percentage=$( bc <<<"scale=4;$passed/$total" )

# Report statistics
echo "Total tests:  ${total}"
echo "Passed tests: ${passed}"
echo "Percentage:   ${percentage}"
