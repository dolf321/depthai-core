# Cleanup in case the script dies
trap 'trap - SIGTERM && kill 0' SIGINT SIGTERM EXIT

# Print timeout if set, otherwise use default
echo "Timeout set to: $1"

# Run USB & PoE stability stress test
DEPTHAI_PROTOCOL=usb ./tests/stability_stress_test $1 &
jobUsb=$!
DEPTHAI_PROTOCOL=tcpip ./tests/stability_stress_test $1 &
jobTcpip=$!

# Wait for tests and save result code
wait $jobUsb || resultUsb=$?
wait $jobTcpip || resultTcpip=$?

# Print results
echo "Stability test USB: $resultUsb"
echo "Stability test PoE: $resultTcpip"

# If both tests concluded successfully, exit with code 0
$(exit $resultUsb) && $(exit $resultTcpip)