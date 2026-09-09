# LighTTice

**Design, simulation, and polarization-resolved analysis of periodic photonic structures.**

LighTTice is a Python-based research toolkit for modelling gratings, photonic crystals, metasurfaces, and other periodic optical structures. It combines three standalone desktop applications into one workflow: geometry design, electromagnetic simulation, and interactive post-processing.

The electromagnetic solver is built around the open-source [`grcwa`](https://grcwa.readthedocs.io/en/latest/#) Rigorous Coupled-Wave Analysis library and adds a graphical workflow for structure definition, material handling, parameter sweeps, parallelized calculations, data export, and polarization-resolved analysis.

## Applications

### LighTTice Designer

`lighttice_designer.py`

Interactive unit-cell editor for periodic structures. It supports circles, ellipses, slits/rectangles, positioning and rotation, material/void regions, periodic previews, and rectangular or non-orthogonal lattice vectors. Designs are exported as JSON cell files for direct use by the Solver.

### LighTTice Solver

`lighttice_solver.py`

GUI-based RCWA simulation application built on `grcwa`. It supports patterned and multilayer structures, dispersive material data, sample/reference calculations, linear-polarization solver runs, diffraction orders, complex modal/field outputs for later polarization reconstruction, configuration save/load, progress estimation, and multiprocessing across energy points. Designer cell files can be loaded directly, including oblique/hexagonal lattice definitions.

### LighTTice Viewer

`lighttice_viewer.py`

Interactive post-processing application for LighTTice NPZ results. It reconstructs reflection, transmission, and absorption using the stored complex modal information and supports zeroth/all propagating orders, sample/reference regions, linear and circular polarization channels, custom Jones states, momentum/energy maps and cuts, persistent computed-map caching, configuration save/load, and a NumPy-based custom-expression workspace for combining calculated signals.

## Installation

Python 3 is required. Install the dependencies with:

```bash
pip install -r requirements.txt
```

`grcwa` documentation: https://grcwa.readthedocs.io/en/latest/#

## Running

Launch each application independently:

```bash
python lighttice_designer.py
python lighttice_solver.py
python lighttice_viewer.py
```

A typical workflow is:

1. Create a periodic unit cell in **LighTTice Designer** and export it as JSON.
2. Load the cell in **LighTTice Solver**, configure the optical stack and simulation range, and save the calculation as NPZ.
3. Open the NPZ in **LighTTice Viewer** for polarization-resolved visualization and custom analysis.

## Main capabilities

- Periodic photonic structures and multilayer stacks
- Rectangular and non-orthogonal/oblique lattice vectors
- Arbitrary unit-cell geometry through the Designer
- RCWA calculations through `grcwa`
- Dispersive material-file support
- Sample and reference simulations
- Parallelized energy sweeps with progress/remaining-time feedback
- Complex reflected/transmitted modal data for post-processing
- Reflection, transmission, and absorption analysis
- Linear, circular, and custom Jones polarization states
- Zeroth-order and all-propagating-order analysis
- Energy–momentum maps and line cuts
- NumPy custom expressions for derived observables
- Persistent NPZ caching of computed maps
- Solver and Viewer configuration save/load

## Notes

LighTTice is research software under active development. Numerical results should be independently validated for the specific geometry, material model, convergence parameters, and observable being studied.

The cell JSON format identifier used internally remains compatible with earlier LighTTice/LighTTice development files.

## Acknowledgement

The RCWA numerical backend used by LighTTice is [`grcwa`](https://grcwa.readthedocs.io/en/latest/#), an open-source differentiable RCWA implementation. Please consult the upstream project documentation for its own usage, citation, and licensing information.
