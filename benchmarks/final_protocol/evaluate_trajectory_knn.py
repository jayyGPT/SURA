#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from benchmarks.knn.wifi_mag_knn import _fit_trajectory_knn
from train.kalmannet_wifiheatmap_magneticCNN_pdr import (
    setup_environment, make_dataset, validate_trajectory_split,
    FUSION_TRAIN_SEED, FUSION_FINAL_TEST_SEED,
)
DB=ROOT/'data/processed/fingerprint_db/it_engineering'; WIFI=ROOT/'checkpoints/wifi_heatmap.pt'; MAG=ROOT/'checkpoints/magnetic_sequence.pt'; OUT=ROOT/'benchmarks/final_protocol/current_results/knn'
regimes={'full':('Full Wi-Fi (1 Hz)',1.0,0.0),'degraded':('Degraded Wi-Fi (5 s, 40% AP drop)',5.0,0.4)}
device=torch.device('cpu'); env=setup_environment(DB,WIFI,MAG,device); OUT.mkdir(parents=True,exist_ok=True); report={}
for key,(name,period,dropout) in regimes.items():
    print('KNN',key,flush=True)
    training=make_dataset(250,FUSION_TRAIN_SEED,env,device,wifi_period_s=period,ap_dropout=dropout,bins=160)
    testing=make_dataset(60,FUSION_FINAL_TEST_SEED,env,device,wifi_period_s=period,ap_dropout=dropout,bins=160)
    audit=validate_trajectory_split(training,testing)
    per_walk,pred,info=_fit_trajectory_knn(training,testing)
    pointwise=np.linalg.norm(pred-testing[6],axis=2)
    d=OUT/key; d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'predictions_and_errors.npz',prediction=pred,target=testing[6],pointwise_error=pointwise,per_walk_mean_error=per_walk)
    report[key]={'name':name,'trajectory_split_audit':audit,**info}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
