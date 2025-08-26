#!/bin/bash

# Script to submit the FirstTests training job to HTCondor
# Usage: ./submit_job.sh

echo "Submitting FirstTests training job..."
JOB_ID=$(condor_submit submit_job.sub | grep -o '[0-9]\+' | head -1)

if [ -n "$JOB_ID" ]; then
    echo "Job submitted with ID: $JOB_ID"
    echo "Check status with: condor_q $JOB_ID"
    echo "Monitor output with: tail -f logs/test_job_${JOB_ID}.0.out"
    echo "Monitor errors with: tail -f logs/test_job_${JOB_ID}.0.err"
    echo "View all logs: ls -la logs/"
else
    echo "Failed to extract job ID. Check status with: condor_q"
    echo "Monitor latest logs with: ls -lt logs/ | head -10"
fi
