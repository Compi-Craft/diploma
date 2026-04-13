#!/bin/bash

while true; do
    echo "[$(date)] Starting port-forward to Prometheus..."
    kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 --address 0.0.0.0 -n monitoring
    echo "[$(date)] Port-forward died. Restarting in 2s..."
    sleep 2
done
