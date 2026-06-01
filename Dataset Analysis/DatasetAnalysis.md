### 🗂️ The Dataset Directory Tree

```text
SURA/Datasets/
|
|-- Magnetic field dataset/
|   |-- Static Data/
|   |   |-- BE Building/         -> Scenario 1 [Phones: A8, G7, S8] [Users: User 2]
|   |   |-- COEX/                -> Scenario 1 [Phones: A8, G6, Q6, S8] [Users: User 2]
|   |   |-- CS Engineering/      -> Scenario 2 [Phones: A8, G7, S8] [Users: User 2]
|   |   |-- Electrical Eng./     -> Scenario 1, 2 [Phones: S8] [Users: User 2]
|   |   |-- IACT/                -> Scenario 1, 2 [Phones: G6, S8] [Users: User 2]
|   |   |-- IT Engineering/      -> Scenario 1, 2, 3 [Phones: S8, A8, G7, G6, Q6] [Users: User 2]
|   |
|   |-- Continuous Data/
|   |   |-- BE Building/         -> Scenario 1 [Phones: A8, G7, S8] [Users: User 2]
|   |   |-- COEX/                -> ❌ (No Data Recorded)
|   |   |-- CS Engineering/      -> Scenario 2 [Phones: A8, G7, S8] [Users: User 1, User 2, User 3]
|   |   |-- Electrical Eng./     -> Scenario 1, 2 [Phones: S8] [Users: User 2]
|   |   |-- IACT/                -> Scenario 1, 2 [Phones: S8] [Users: User 2]
|   |   |-- IT Engineering/      -> Scenario 1 [Phones: A8, G6, Q6, S8, S9+] [Users: User 1, 2, 3, 4]
|
|-- WiFi dataset/
    |-- BE Engineering/          -> Scenario 1 [Phones: A8, G7, S8] [Users: User 2]
    |-- COEX /                   -> Scenario 1 [Phones: A8, G6, Q6, S8] [Users: User 2]
    |-- CS Engineering/          -> Scenario 2 [Phones: A8, G7, S8] [Users: User 2]
    |-- Electrical Eng./         -> Scenario 2 [Phones: A8, G7, S8] [Users: User 2]
    |-- IACT/                    -> Scenario 2 [Phones: A8, G7, S8] [Users: User 2]
    |-- IT Engineering/          -> Scenario 1, 2, 3 [Phones: S8] [Users: User 2]
```

---

### 📊 Unified Dataset Overview Table

> **Note:** Data below was verified against the actual filesystem. Some discrepancies were found with the original tree (marked with ⚠️ in notes below the table). WiFi `.csv` files are actually **binary `.xls`** (Excel BIFF) format despite the `.csv` extension.

<table>
  <tr>
    <th rowspan="3" style="text-align:center;">Building</th>
    <th colspan="8" style="text-align:center;">📡 Magnetic Field Dataset</th>
    <th colspan="4" style="text-align:center;">📶 WiFi Dataset</th>
    <th rowspan="3" style="text-align:center;">Modes<br/>(Holding Styles)</th>
    <th rowspan="3" style="text-align:center;">Magnetic<br/>Sensor Columns</th>
  </tr>
  <tr>
    <th colspan="4" style="text-align:center;">Static Data</th>
    <th colspan="4" style="text-align:center;">Continuous Data</th>
    <th colspan="4" style="text-align:center;">&nbsp;</th>
  </tr>
  <tr>
    <th>Scenarios</th><th># Scn</th><th>Phones</th><th>Users</th>
    <th>Scenarios</th><th># Scn</th><th>Phones</th><th>Users</th>
    <th>Scenarios</th><th># Scn</th><th>Phones</th><th>Users</th>
  </tr>
  <!-- BE Building -->
  <tr>
    <td><strong>BE Building</strong></td>
    <td>1</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>1</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>1</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>Navigation</td>
    <td rowspan="6">Time, X-cord, Y-cord,<br/>Mag_x, Mag_y, Mag_z,<br/>Acc_x, Acc_y, Acc_z,<br/>Gyro_x, Gyro_y, Gyro_z,<br/>Orn_x, Orn_y, Orn_z,<br/>Pressure</td>
  </tr>
  <!-- COEX -->
  <tr>
    <td><strong>COEX (Mall)</strong></td>
    <td>1</td><td>1</td><td>G6, S8</td><td>User 2</td>
    <td colspan="4" style="text-align:center;">❌ No Data Recorded</td>
    <td>1</td><td>1</td><td>G6, S8</td><td>User 2</td>
    <td>Navigation</td>
  </tr>
  <!-- CS Engineering -->
  <tr>
    <td><strong>CS Engineering</strong></td>
    <td>2</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>2</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 1, 2, 3</td>
    <td>2</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>Navigation</td>
  </tr>
  <!-- Electrical Eng. -->
  <tr>
    <td><strong>Electrical Eng.</strong></td>
    <td>1, 2</td><td>2</td><td>S1: S8 · S2: A8, G7, S8, S9+</td><td>User 2</td>
    <td>1, 2</td><td>2</td><td>S1: S8 · S2: A8, G7, S8, S9+</td><td>User 2</td>
    <td>2</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>Navigation</td>
  </tr>
  <!-- IACT -->
  <tr>
    <td><strong>IACT</strong></td>
    <td>1, 2</td><td>2</td><td>S1: G6, S8 · S2: A8, G7, S8, S9+</td><td>User 2</td>
    <td>1, 2</td><td>2</td><td>S1: S8 · S2: A8, G7, S8, S9+</td><td>User 2</td>
    <td>2</td><td>1</td><td>A8, G7, S8, S9+</td><td>User 2</td>
    <td>Navigation</td>
  </tr>
  <!-- IT Engineering -->
  <tr>
    <td><strong>IT Engineering</strong></td>
    <td>1, 2, 3</td><td>3</td><td>S1: A8, G7, S8, S9+ · S2: A8, G7, S8, S9+ · S3: G6, Q6, S8</td><td>User 2</td>
    <td>1, 2</td><td>2</td><td>S1: A8, G6, Q6, S8 · S2: A8, G7, S8, S9+</td><td>S1: User 1, 2, 3 · S2: User 2</td>
    <td>1, 2, 3</td><td>3</td><td>S1: A8, G7, S8, S9+ · S2: A8, G7, S8, S9+ · S3: S8</td><td>User 2</td>
    <td>Navigation, Call listening, Swinging, Room, Stairs</td>
  </tr>
</table>

#### ⚠️ Filesystem Discrepancies vs. Original Tree
> The following differences were found by verifying the actual directory structure:
> - **BE Building:** Filesystem has **S9+** in Static, Continuous, and WiFi — original tree listed only `A8, G7, S8`.
> - **COEX:** Filesystem shows only **G6, S8** (not A8, Q6) for both Static Magnetic and WiFi.
> - **CS Engineering:** Filesystem has **S9+** across all three data types — original tree did not list it.
> - **Electrical Eng.:** Scenario 2 in the filesystem has **A8, G7, S8, S9+** — original tree listed only `S8`.
> - **IACT:** Scenario 2 in the filesystem has **A8, G7, S8, S9+** — original tree listed only `G6, S8`.
> - **IT Engineering (Continuous):** Filesystem shows **Scenario 1 AND Scenario 2** — original tree listed only `Scenario 1`. IT Eng Continuous S1 users vary by phone: A8→User 1,3 / G6→User 2 / Q6→User 2 / S8→User 1,2,3.
> - **WiFi (multiple buildings):** Filesystem shows **A8, G7, S8, S9+** for EE, IACT, CS, IT (S1/S2) — original tree had different/fewer phone lists.

---

### Context & Analysis
#### 1. Data Types ?
*   **Static Data (Magnetic):** The surveyors gathered this data by walking 1 meter, stopping entirely, placing their feet exactly on a grid point marked on the floor, and recording several seconds of stable magnetic sensor reading. *This provides highly accurate ground truth.* 
*   **Continuous Data (Magnetic):** The surveyors pressed "record" and walked continuously down the corridor without stopping at the individual nodes. *This evaluates if an ML model can track raw trajectory patterns through movement noise.*
*   **WiFi Data:** Taken identically to the Static approach, standing on nodes and scanning ambient Access Points (APs).

#### 2. Buildings and Scenarios ?
*   **Buildings:** The data was recorded in the halls of an academic campus and one shopping mall. Notably, **COEX (Mall)** has no Continuous data available—the researchers simply did not run walking tests there.
*   **Scenarios:** A Scenario defines the **physical walking trajectory** chosen on that building's floor plan. `Scenario 1` usually indicates walking in a simple rectangular loop around the central corridors. `Scenario 2` or `3` means moving through alternate pathways, side corridors, or reversing directions to record the identical hall but facing a different magnetic field direction.

#### 3. Phones ?
**Device Heterogeneity**. A Samsung Galaxy S8 has wildly different sensor calibration, noise, and Wi-Fi antenna strength than an LG G6.

#### 4. Users ?
Evaluating **Spatial Diversity (User Gait)**. People have different walking speeds, phone-holding habits, arm swinging, and heights (which changes the distance of the magnetometer to the steel beams in the floor). 
*   **User 2 `(M-174cm)`:** This was the primary surveyor. As you see in the tree above, nearly **100%** of the static mapping architectures (Static & WiFi) were generated solely by this 174cm male.
*   **Users 1, 3, 4:** These participants were used almost exclusively for **Continuous** recording. The authors had them walk the paths normally to see if an ML model trained on User 2's dataset would break when User 1 `(F-160cm)` walked the exact same route with a shorter stride length. 

---

### 🔍 Deep Dive Analysis

We performed an exhaustive scan of the entire dataset, parsing **all 4,261 magnetic field CSV files** row-by-row and cataloging all **4,399 WiFi binary files** to extract precise ground-truth statistics. 

> [!TIP]
> **View the full report:** For the complete breakdown including exact null counts, sensor ranges, coordinate spaces, and anomalies, see the full artifact: [dataset_deep_dive.md](file:///C:/Users/lenovo/.gemini/antigravity-ide/brain/f649f5bc-f8a8-408e-b9d7-415aabf882c9/dataset_deep_dive.md)

**Key Observations:**
1. **COEX Corruption:** All 451 files for COEX are entirely broken (binary `.xls` files misnamed as `.csv`), causing ~100K null rows. This is the source of the 14.5% global null rate.
2. **Missing Sensor Columns:** The `Orn_z` and `Pressure` columns are systematically missing (~35% null rate) across all buildings except COEX, likely due to older recordings or unsupported phone models.
