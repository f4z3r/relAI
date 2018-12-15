#!/usr/bin/env bash

# If single example is launched, launch that
if [[ $# -eq 3 ]]; then
    python3 analyzer.py ../mnist_nets/$1.txt ../mnist_images/img$2.txt $3
    exit $?
fi

# Test single network with specified epsilon over all images
if [[ $# -eq 2 ]]; then
    echo "Testing net '$1' for epsilon $2"
    # Declare variables
    declare -i total=0
    declare -i passed=0

    net="$1"
    eps="$2"

    # Start timer
    start=`date +%s`

    for img_num in {0..99}; do
        ((++total))
        echo -n "Testing image $img_num ... "
        iter_start=`date +%s`

        # Run the analyzer
        python3 analyzer.py ../mnist_nets/$net.txt ../mnist_images/img$img_num.txt $eps > /dev/null
        return_code=$?

        iter_stop=`date +%s`
        iter_time=$((iter_stop-iter_start))

        # Check return code and increase if passed
        if [[ $return_code -eq 0 ]]; then
            echo "verified in ${iter_time}s"
            ((++passed))
        else
            echo "FAILED in ${iter_time}s"
        fi
    done

    # Stop timer
    end=`date +%s`
    runtime=$((end-start))

    # Compute passed percentage
    percentage=$( bc <<<"scale=2;(${passed}/${total})*100" )
    # Compute time per test
    time_per_test=$( bc <<<"scale=2;${runtime}/${total}" )

    # Report statistics
    echo "Total tests:  ${total}"
    echo "Passed tests: ${passed}"
    echo "Percentage:   ${percentage}%"
    echo "Total time:   ${runtime}s"
    echo "Time/test:    ${time_per_test}s"
fi
