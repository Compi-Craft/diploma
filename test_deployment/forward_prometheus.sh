#!/bin/bash

kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 --address 0.0.0.0
