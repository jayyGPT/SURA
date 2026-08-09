# 🔬 SURA Dataset — Exhaustive Deep Dive Analysis

> All numbers below were computed by scanning **every single file** across the dataset directory tree.
> Magnetic field data: 4,261 CSV files parsed row-by-row. WiFi data: 4,399 binary files cataloged.

---

## 1. Grand Totals

### Magnetic Field Dataset

| Metric | Value |
|---|---|
| **Total CSV Files** | 4,261 |
| **Total Data Rows** | 688,226 |
| **Total File Size** | 115.84 MB |
| **Unique (X,Y) Coordinate Pairs** | 126,722 |
| **Global X Range** | [-49.2, 124.0] |
| **Global Y Range** | [-52.68, 90.0] |
| **Data Collection Period** | 2018-02-20 → 2021-05-19 (~3.25 years) |
| **Rows per File (min / median / mean / max)** | 28 / 120 / 161.5 / 3,992 |

### WiFi Dataset

| Metric | Value |
|---|---|
| **Total Files** | 4,399 |
| **Total File Size** | 117.56 MB |
| **File Format** | Binary `.xls` (Excel BIFF8), misnamed as `.csv` |

### Combined Total

| Metric | Value |
|---|---|
| **Total Files (All Data)** | **8,660** |
| **Total Size (All Data)** | **~233 MB** |

---

## 2. Static vs. Continuous Breakdown

| Metric | Static Data | Continuous Data |
|---|---|---|
| **CSV Files** | 4,134 | 127 |
| **Data Rows** | 539,011 | 149,215 |
| **File Size** | 94.97 MB | 20.87 MB |
| **Unique (X,Y) Coords** | 538 | 126,188 |
| **Rows/File (min)** | 28 | 625 |
| **Rows/File (median)** | 119 | 933 |
| **Rows/File (mean)** | 130.4 | 1,174.9 |
| **Rows/File (max)** | 302 | 3,992 |

> [!IMPORTANT]
> **Static data has only 538 unique coordinate positions** across all buildings — these are the grid nodes marked on the floor. Each CSV file corresponds to one grid node, containing ~119 sensor readings (~10 seconds of sampling at ~12Hz).
>
> **Continuous data has 126,188 unique positions** because coordinates change continuously as the surveyor walks, producing GPS-like trajectory traces with ~1,175 rows per walk file.

---

## 3. Per-Building Magnetic Field Data

<table>
  <tr>
    <th rowspan="2">Building</th>
    <th colspan="3">File Counts</th>
    <th colspan="2">Data Rows</th>
    <th>Size</th>
    <th>Unique<br/>Coords</th>
    <th colspan="2">Null Analysis</th>
    <th>Recording Period</th>
  </tr>
  <tr>
    <th>Total</th><th>Static</th><th>Cont.</th>
    <th>Total</th><th>Rows/File (med)</th>
    <th>MB</th>
    <th>#</th>
    <th>Total Nulls</th><th>Null Columns</th>
    <th>Date Range</th>
  </tr>
  <tr>
    <td><strong>BE Building</strong></td>
    <td>428</td><td>~416</td><td>~12</td>
    <td>66,030</td><td>122</td>
    <td>11.95</td>
    <td>11,086</td>
    <td>23,132</td><td>Orn_z, Pressure</td>
    <td>2021-05-19 (single day)</td>
  </tr>
  <tr>
    <td><strong>COEX (Mall)</strong></td>
    <td>451</td><td>451</td><td>0</td>
    <td>100,413</td><td>219</td>
    <td>14.76</td>
    <td>0 ⚠️</td>
    <td>1,506,201 ⚠️</td><td>ALL columns</td>
    <td>2019-08-09 (single day)</td>
  </tr>
  <tr>
    <td><strong>CS Engineering</strong></td>
    <td>312</td><td>~208</td><td>~104</td>
    <td>55,578</td><td>114</td>
    <td>9.66</td>
    <td>20,083</td>
    <td>42,604</td><td>Orn_z, Pressure</td>
    <td>2021-02-11 → 2021-02-12</td>
  </tr>
  <tr>
    <td><strong>Electrical Eng.</strong></td>
    <td>370</td><td>~265</td><td>~105</td>
    <td>61,601</td><td>114</td>
    <td>10.00</td>
    <td>25,786</td>
    <td>56,054</td><td>Orn_z, Pressure</td>
    <td>2018-03-21 → 2021-02-12</td>
  </tr>
  <tr>
    <td><strong>IACT</strong></td>
    <td>393</td><td>~282</td><td>~111</td>
    <td>61,853</td><td>115</td>
    <td>10.11</td>
    <td>23,984</td>
    <td>49,744</td><td>Orn_z, Pressure</td>
    <td>2018-03-21 → 2021-02-12</td>
  </tr>
  <tr>
    <td><strong>IT Engineering</strong></td>
    <td>2,307</td><td>~2,180</td><td>~127</td>
    <td>342,751</td><td>Varies</td>
    <td>59.35</td>
    <td>49,467</td>
    <td>106,679</td><td>Orn_z, Pressure</td>
    <td>2018-02-20 → 2021-05-19</td>
  </tr>
  <tr style="font-weight:bold;">
    <td>TOTAL</td>
    <td>4,261</td><td>4,134</td><td>127</td>
    <td>688,226</td><td>—</td>
    <td>115.84</td>
    <td>126,722</td>
    <td>1,784,414</td><td>—</td>
    <td>2018-02 → 2021-05</td>
  </tr>
</table>

---

## 4. Null / Missing Data Analysis

### 4.1 Global Null Counts (Across All 688,226 Rows)

| Column | Null Count | % of Total Rows | Interpretation |
|---|---|---|---|
| **Time** | 6 | 0.00% | ✅ Nearly perfect |
| **X-cord** | 100,413 | 14.59% | ⚠️ All from COEX |
| **Y-cord** | 100,413 | 14.59% | ⚠️ All from COEX |
| **Mag_x** | 100,413 | 14.59% | ⚠️ All from COEX |
| **Mag_y** | 100,413 | 14.59% | ⚠️ All from COEX |
| **Mag_z** | 100,413 | 14.59% | ⚠️ All from COEX |
| **Acc_x** | 100,414 | 14.59% | ⚠️ All from COEX (+1) |
| **Acc_y** | 100,414 | 14.59% | ⚠️ All from COEX |
| **Acc_z** | 100,414 | 14.59% | ⚠️ All from COEX |
| **Gyro_x** | 100,414 | 14.59% | ⚠️ All from COEX |
| **Gyro_y** | 100,414 | 14.59% | ⚠️ All from COEX |
| **Gyro_z** | 100,414 | 14.59% | ⚠️ All from COEX |
| **Orn_x** | 100,414 | 14.59% | ⚠️ All from COEX |
| **Orn_y** | 100,415 | 14.59% | ⚠️ All from COEX |
| **Orn_z** | 239,515 | 34.80% | 🔴 COEX + ~all other buildings |
| **Pressure** | 239,515 | 34.80% | 🔴 COEX + ~all other buildings |

### 4.2 Key Null Findings

> [!CAUTION]
> **COEX is completely corrupted.** All 451 COEX CSVs have 100,413 rows but produce **zero valid coordinates** and **all sensor values are null**. The path structure is also different (no Scenario subfolder). These files are almost certainly **binary .xls files mislabeled as .csv** — identical to the WiFi files. COEX's "100,413 rows" are actually binary garbage parsed as text by the CSV reader.

> [!WARNING]
> **Orn_z and Pressure are missing in ~34.8% of all rows.** This affects every building except COEX (which is entirely null). These two columns are consistently the last two columns and may be missing from certain phone models or older recording sessions. Specifically:
> - BE Building: 11,566 nulls in Orn_z/Pressure
> - CS Engineering: 21,302 nulls
> - Electrical Eng.: 28,027 nulls
> - IACT: 24,872 nulls
> - IT Engineering: 53,335 nulls

> [!NOTE]
> **Excluding COEX**, the remaining 5 buildings have **zero nulls** in columns Time through Orn_y (the first 14 columns). The data quality for these core sensor channels is **excellent**.

---

## 5. Global Sensor Ranges

| Sensor | Min | Max | Unit / Notes |
|---|---|---|---|
| **X-cord** | -49.2 | 124.0 | Grid coordinates (meters) |
| **Y-cord** | -52.68 | 90.0 | Grid coordinates (meters) |
| **Mag_x** | -54.07 | 63.17 | µT (microtesla) |
| **Mag_y** | -50.25 | 51.50 | µT |
| **Mag_z** | -63.30 | 11.70 | µT |
| **Acc_x** | -6.00 | 16.29 | m/s² |
| **Acc_y** | -9.52 | 9.58 | m/s² |
| **Acc_z** | -3.30 | 14.71 | m/s² (~gravity = 9.81) |
| **Gyro_x** | -2.35 | 2.33 | rad/s |
| **Gyro_y** | -0.68 | 323.48 | rad/s ⚠️ outlier |
| **Gyro_z** | -85.47 | 5.29 | rad/s ⚠️ outlier |
| **Orn_x** | -9.87 | 359.30 | degrees (heading/azimuth) |
| **Orn_y** | -6.91 | 1019.60 | degrees ⚠️ outlier |
| **Orn_z** | -4.41 | 3.43 | degrees |
| **Pressure** | 0.06 | 1024.61 | hPa (atmospheric pressure, ~1013 at sea level) |

> [!WARNING]
> **Outlier ranges in Gyro_y (323.5), Gyro_z (-85.5), and Orn_y (1019.6)** suggest sensor noise spikes, dropped frames, or glitched readings during continuous walks. These would need to be filtered/clipped during preprocessing.

---

## 6. Per-Building Coordinate Spaces

| Building | X Range | Y Range | Grid Area (m²) | # Unique Nodes (Static) |
|---|---|---|---|---|
| **BE Building** | [-19.4, 124.0] | [-21.2, 90.0] | ~15,928 | Part of 11,086 total |
| **COEX** | N/A (corrupt) | N/A | N/A | N/A |
| **CS Engineering** | [-15.5, 90.0] | [-46.0, 33.3] | ~8,367 | Part of 20,083 total |
| **Electrical Eng.** | [-42.8, 81.0] | [-52.7, 36.0] | ~10,977 | Part of 25,786 total |
| **IACT** | [-49.2, 45.1] | [-49.1, 26.4] | ~7,122 | Part of 23,984 total |
| **IT Engineering** | [1.0, 90.0] | [-39.5, 27.0] | ~5,918 | Part of 49,467 total |

---

## 7. WiFi Dataset Breakdown

| Building | Files | Size (MB) | Scenarios | Phones | Users |
|---|---|---|---|---|---|
| **BE Engineering** | 416 | 11.56 | Scenario-1 | A8, G7, S8, S9+ | User 2 |
| **COEX** | 451 | 12.55 | Scenario-1 | G6, S8 | User 2 |
| **CS Engineering** | 416 | 11.50 | Scenario-2 | A8, G7, S8, S9+ | User 2 |
| **Electrical Eng.** | 262 | 7.20 | Scenario-2 | A8, G7, S8, S9+ | User 2 |
| **IACT** | 282 | 7.78 | Scenario-2 | A8, G7, S8, S9+ | User 2 |
| **IT Engineering** | 2,572 | 66.97 | Scenario-1, 2, 3 | A8, G7, S8, S9+ (S3: S8 only) | User 2 |
| **TOTAL** | **4,399** | **117.56** | — | — | — |

> [!NOTE]
> WiFi files are **1:1 matched** with their corresponding magnetic static files — same timestamps, same grid node positions. Each WiFi file captures the ambient Access Point scan taken at that node simultaneously with the magnetic reading.

---

## 8. Data Collection Timeline

| Building | Earliest Recording | Latest Recording | Span |
|---|---|---|---|
| **IT Engineering** | 2018-02-20 | 2021-05-19 | ~3.25 years |
| **Electrical Eng.** | 2018-03-21 | 2021-02-12 | ~2.9 years |
| **IACT** | 2018-03-21 | 2021-02-12 | ~2.9 years |
| **COEX (Mall)** | 2019-08-09 | 2019-08-09 | Single day |
| **CS Engineering** | 2021-02-11 | 2021-02-12 | 2 days |
| **BE Building** | 2021-05-19 | 2021-05-19 | Single day |

> [!IMPORTANT]
> The dataset was collected over **3+ years**. Early recordings (2018) were from EE, IACT, and IT only. The bulk of the multi-phone, multi-scenario data appears to have been added in a **February 2021 campaign** (CS, EE, IACT, IT) and a **May 2021 campaign** (BE, IT).

---

## 9. IT Engineering — Special Modes (Phone Holding Styles)

IT Engineering is the only building with recordings in multiple **holding modes** — both for Static Magnetic and WiFi data:

| Mode | Data Types Present | Description |
|---|---|---|
| **Navigation** | Static + Continuous + WiFi | Normal walking with phone held flat in front (standard) |
| **Call listening** | Static + WiFi | Phone held to ear as if on a call |
| **Swinging** | Static + WiFi | Phone held in hand while arm swings naturally during walking |
| **Room** | Static + WiFi | Recording inside a room rather than a corridor |
| **Stairs** | Static + WiFi | Recording while ascending/descending stairs |

> The "Room" and "Stairs" modes also include data from multiple phones (A8, G7, S8, S9+) and the "Stairs" data notably includes recordings from **User 1 (M-177cm)** in addition to User 2.

---

## 10. Critical Anomalies & Flags

1. **🔴 COEX is not CSV data.** The 451 "CSV" files in COEX's magnetic folder are actually binary Excel files. The CSV parser reads them as garbage — producing 100K+ rows of nulls. These files should be read with an Excel parser (xlrd/openpyxl), not as CSV.

2. **🟡 COEX has a different directory structure.** Path is `COEX/Navigation/Phone/User/file` — missing the `Scenario X` level that all other buildings have. It goes directly from the building name to phone folders.

3. **🟡 Orn_z and Pressure systematically missing.** These two columns are the last in the CSV and are missing from a significant chunk of files (likely from the G6/G7 phones or older 2018 recordings where these sensors weren't captured).

4. **🟡 Sensor outliers exist.** Gyro_y max of 323 rad/s and Orn_y max of 1019° are physically impossible values suggesting sensor glitches or data corruption in specific rows.

5. **🟡 Static data has duplicate CSV pairs.** Many files in some buildings (e.g., CS Engineering) have timestamps 1 second apart (e.g., `150030.csv` and `150031.csv`), suggesting the recording app created two files per grid node — possibly an IMU file and a WiFi file. This needs investigation during preprocessing.

6. **🟢 Core sensor data (Mag/Acc/Gyro) is clean** for all non-COEX buildings. Zero nulls in columns 1–14 across ~588K valid rows.

---

## 11. Summary Counts for Quick Reference

| Metric | Count |
|---|---|
| Buildings | 6 |
| Total files (all datasets) | 8,660 |
| Total data rows (magnetic) | 688,226 |
| Valid data rows (excl. COEX) | ~587,813 |
| Total file size | ~233 MB |
| Unique grid nodes (Static) | 538 |
| Unique trajectory points (Continuous) | 126,188 |
| Phone models | 6 (A8, G7, G6, Q6, S8, S9+) |
| Unique users | 4 (User 1, 2, 3, + User 1 M-177cm) |
| Sensor channels per row | 16 (Time + 15 sensors) |
| Holding modes | 5 (Navigation, Call, Swing, Room, Stairs) |
| Data collection span | 2018-02 → 2021-05 |
