# Mejias et al. 2016 - Large-Scale Cortical Network Simulation

Overview of the main results of the simulation in Python. This implementation includes the full largescale connection model as well as adding lesions analysis.

## Data Organization

All simulation results and plots are saved in the `data/` folder, organized by analysis type:

```
data/
├── intralaminar/
├── interlaminar_a/
├── interlaminar_b/
├── interlaminar_c/
├── interareal/
├── largescale/
└── lesion_<area_names>/    # e.g., lesion_V4, lesion_8l_7A_46d
```

## Intralaminar Analysis

This model considers the simulation of excitatory and inhibitory neurons of supra- and infragranular layers.

**Run the analysis:**
```bash
python main.py -analysis intralaminar
```

**Output files (saved to `data/intralaminar/`):**
- `L23_simulation.pckl` - Simulation data
- `intralaminar_2B.png` - Power spectrum analysis
- `intralaminar_2C.png` - Peak power and frequency analysis

## Interlaminar Analysis

Here we consider connections between the L2/3 and L5/6 layer.

### Interlaminar A - Power Spectrum Comparison

**Run the analysis:**
```bash
python main.py -analysis interlaminar_a
```

**Output files (saved to `data/interlaminar_a/`):**
- `simulation.pckl` - Simulation data
- `power_spectrum.png` - Coupled vs uncoupled power spectrum
- `power_spectrum_neurodsp_L23.png` - L2/3 power spectrum (neurodsp)
- `power_spectrum_neurodsp_L56.png` - L5/6 power spectrum (neurodsp)

### Interlaminar B - Activity Traces

To analyse the dynamic interaction between the two layers, this analyses the LFP (Local Field Potential) from the layer 5/6 with respect to the alpha rhythm from Layer 2/3.

**Run the analysis:**
```bash
python main.py -analysis interlaminar_b
```

**Output files (saved to `data/interlaminar_b/`):**
- `simulation.pckl` - Simulation data
- `activity_traces.png` - L5/6 activity traces aligned to alpha peaks

### Interlaminar C - Input Sweep Analysis

Recreates figure 3C from the original paper.

**Run the analysis:**
```bash
python main.py -analysis interlaminar_c
```

**Output files (saved to `data/interlaminar_c/`):**
- `interlaminar_3C.png` - Multi-panel analysis of gamma/alpha power vs L5/6E input

## Interareal Analysis (Figure 4)

Simulates two cortical areas (V1 and V4) with interareal connectivity to study how stimulation in one area affects oscillatory dynamics in the other.

**Run the analysis:**
```bash
python main.py -analysis interareal
```

**Output files (saved to `data/interareal/`):**
- `stimulate_V1_layer_V4_l23.png` - V1 stimulation, V4 L2/3 response
- `stimulate_V1_layer_V4_l56.png` - V1 stimulation, V4 L5/6 response
- `stimulate_V1_layer_gamma.png` - Gamma power comparison
- `stimulate_V1_layer_alpha.png` - Alpha power comparison
- `stimulate_V4_layer_V1_l23.png` - V4 stimulation, V1 L2/3 response
- `stimulate_V4_layer_V1_l56.png` - V4 stimulation, V1 L5/6 response
- `stimulate_V4_layer_gamma.png` - Gamma power comparison
- `stimulate_V4_layer_alpha.png` - Alpha power comparison

## Large-Scale Network Analysis

Simulates a full 30-area macaque cortical network with realistic connectivity based on anatomical data. Analyzes oscillatory power (alpha and gamma bands) across all areas.

**Run the analysis:**
```bash
python main.py -analysis largescale
```

**Output files (saved to `data/largescale/`):**
- `simulation.pckl` - Full network simulation data
- `power_by_area.png` - Alpha and gamma power by area

## Lesion Analysis (Stroke Simulation)

Simulates cortical lesions (strokes) by selectively disabling brain areas and measuring the impact on network dynamics. This allows studying:
- How lesions affect firing rates in other areas
- Changes in oscillatory power (alpha/gamma bands)
- Distance-dependent effects of lesions
- Network resilience and compensatory mechanisms

**Run the analysis:**
```bash
python main.py -analysis lesion
```

**Configuration (in `main.py`):**
- `lesion_areas_names`: List of areas to lesion (e.g., `['V4']` or `['8l', '7A', '46d']`)
- `lesion_type`: Type of lesion
  - `'complete'`: Remove all activity and connectivity (default)
  - `'activity_only'`: Clamp activity to 0, keep connections
  - `'output_loss'`: Remove outgoing connections only
  - `'input_loss'`: Remove incoming connections only

**Output files (saved to `data/lesion_<area_names>/`):**
- `baseline_simulation.pckl` - Healthy network simulation
- `lesion_simulation.pckl` - Lesioned network simulation
- `lesion_1_rate_changes.png` - Firing rate changes per area (excluding lesioned areas)
- `lesion_2_absolute_rates.png` - Baseline vs lesioned absolute rates
- `lesion_3_gamma_power.png` - Gamma band power changes
- `lesion_4_alpha_power.png` - Alpha band power changes
- `lesion_5_distance_effects.png` - Distance-dependent effects scatter plot
- `power_by_area.png` - Power analysis for lesioned network

**Example:**
- Lesioning V4: Results saved to `data/lesion_V4/`
- Lesioning multiple areas (8l, 7A, 46d): Results saved to `data/lesion_8l_7A_46d/`

## Available Analysis Types

- `intralaminar` - Single-layer dynamics
- `interlaminar_a` - Power spectrum comparison
- `interlaminar_b` - Activity traces
- `interlaminar_c` - Input sweep analysis
- `interareal` - Two-area connectivity
- `largescale` - Full 30-area network
- `lesion` - Stroke/lesion simulation
- `debug` - Debugging mode

## Notes

- All simulations use a default time step of `dt=2e-4` seconds (0.2 ms)
- Results are automatically saved to prevent re-computation
- The `data/` folder structure keeps all outputs organized
- Lesion analysis folder names automatically reflect the lesioned areas
