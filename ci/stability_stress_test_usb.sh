# Print timeout if set, otherwise use default
echo "Timeout set to: $1"

# Run USB stability stress test
DEPTHAI_PROTOCOL=usb ./tests/stability_stress_test $1
