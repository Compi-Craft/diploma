#!/bin/bash

while true; do
    echo "[$(date)] Starting port-forward to cpu-service..."
    kubectl port-forward svc/cpu-service 8080:8080
    echo "[$(date)] Port-forward died. Restarting in 2s..."
    sleep 2
done
