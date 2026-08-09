# System overview

The target architecture separates spatial inference, relative motion, and temporal fusion.

```text
Wi-Fi RSSI ───────► Wi-Fi heatmap MLP ───────► z_wifi, uncertainty
                                                     │
IMU ──────────────► causal PDR ───────────────► u_t  │
                                                     ├──► DualKalmanNet ─► x_t
Magnetic sequence ► 1D CNN ───────────────────► z_mag, uncertainty
```

The checked-in legacy baseline still contains a scalar anomaly-gradient magnetic update. It is retained only so the previous paper values remain reproducible. The approved next architecture removes that anomaly map and feeds the magnetic CNN output directly into DualKalmanNet.

All three measurement streams must remain causal. Sensor masks must make every unavailable-modality configuration explicit during both training and evaluation.
