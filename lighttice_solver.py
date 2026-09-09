#!/usr/bin/env python3
"""
lighTTice Solver — PyQt5/QtAgg RCWA simulation application for periodic photonic structures.

This version saves complex reflected modal amplitudes b0 and transmitted modal amplitudes aN for robust Jones/circular-polarization postprocessing, in addition to field outputs. It also supports arbitrary homogeneous top layers and full configuration save/load. Each top layer can use:
    - constant refractive index n + i*k
    - constant epsilon eps_real + i*eps_imag
    - a spectral material file with a friendly preview / plot / column-mapping dialog

Requirements:
    pip install grcwa numpy matplotlib scipy PyQt5

Run:
    python rcwa_gold_slit_gui_qtagg_pyqt5_layers.py

Outputs shown:
    - Reflection R for the slit array
    - Delta R/R = (R_main - R_ref) / R_ref

Reference stack is user selectable:
    - Uniform gold: air / selected reference top layers / uniform Au film / glass
    - Patterned gold: air / selected reference top layers / slitted Au / glass
    - No gold: air / selected reference top layers / glass

For each top homogeneous layer, choose whether it is included in the reference.
This lets you compute, for example:
    R_pattern: air / layer A / layer B / slitted Au / glass
    R_ref:     air / layer A / uniform Au / glass

Units:
    - Geometry: microns
    - kx, ky: um^-1
    - Photon energy: eV
"""

import sys
import csv
import json
import cmath
import traceback
import time
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QSplitter, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox, QCheckBox,
    QFileDialog, QMessageBox, QProgressBar, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox
)

try:
    import grcwa
except ImportError:
    grcwa = None


HC_EV_UM = 1.239841984
HC_EV_NM = 1239.841984
SOLVER_VERSION = "0.5-oblique-lattice-physical-preview"
CONFIG_VERSION = "2.0-direct-CP"


@dataclass
class Material:
    mode: str = "n"  # "n", "eps", or "file"
    n_value: complex = 1.0 + 0.0j
    eps_value: complex = 1.0 + 0.0j
    file_path: str = ""
    file_x_kind: str = "energy_ev"
    file_data_kind: str = "nk"  # "nk" or "epsilon"
    file_x_col: int = 0
    file_n_col: int = 1
    file_k_col: int = 2
    file_eps_re_col: int = 1
    file_eps_im_col: int = 2
    file_delimiter: str = "auto"
    _cache: Optional[tuple] = field(default=None, init=False, repr=False)

    def epsilon(self, energy_ev: float) -> complex:
        if self.mode == "n":
            return self.n_value ** 2
        if self.mode == "eps":
            return self.eps_value
        if self.mode == "file":
            E, n, k, eps_re, eps_im = self.load_arrays()
            order = np.argsort(E)
            E = E[order]
            if self.file_data_kind == "epsilon":
                er = np.interp(energy_ev, E, eps_re[order])
                ei = np.interp(energy_ev, E, eps_im[order])
                return complex(er, ei)
            nr = np.interp(energy_ev, E, n[order])
            ki = np.interp(energy_ev, E, k[order])
            return complex(nr, ki) ** 2
        raise ValueError(f"Unknown material mode: {self.mode}")

    def split_line(self, line: str):
        line = line.strip()
        if not line:
            return []
        if self.file_delimiter == "comma":
            return [x.strip() for x in line.split(",")]
        if self.file_delimiter == "tab":
            return [x.strip() for x in line.split("\t")]
        if self.file_delimiter == "space":
            return line.split()
        return [x.strip() for x in line.split(",")] if "," in line else line.split()

    def load_arrays(self):
        if self._cache is not None:
            return self._cache
        if not self.file_path:
            raise ValueError("File material selected, but no file path is set.")

        max_col = max(
            self.file_x_col, self.file_n_col, self.file_k_col,
            self.file_eps_re_col, self.file_eps_im_col
        )

        energies = []
        n_values = []
        k_values = []
        eps_re_values = []
        eps_im_values = []

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self.split_line(line)
                if len(parts) <= max_col:
                    continue
                try:
                    x = float(parts[self.file_x_col])
                    if self.file_x_kind == "energy_ev":
                        E = x
                    elif self.file_x_kind == "wavelength_nm":
                        E = HC_EV_NM / x
                    elif self.file_x_kind == "wavelength_um":
                        E = HC_EV_UM / x
                    elif self.file_x_kind == "wavelength_m":
                        E = HC_EV_UM / (x * 1e6)
                    else:
                        raise ValueError(f"Unknown file_x_kind: {self.file_x_kind}")

                    if self.file_data_kind == "epsilon":
                        er = float(parts[self.file_eps_re_col])
                        ei = float(parts[self.file_eps_im_col])
                        n_complex = cmath.sqrt(complex(er, ei))
                        nr = float(np.real(n_complex))
                        ki = float(np.imag(n_complex))
                        if ki < 0:
                            nr = -nr
                            ki = -ki
                    else:
                        nr = float(parts[self.file_n_col])
                        ki = float(parts[self.file_k_col])
                        eps_c = complex(nr, ki) ** 2
                        er = float(np.real(eps_c))
                        ei = float(np.imag(eps_c))

                    energies.append(E)
                    n_values.append(nr)
                    k_values.append(ki)
                    eps_re_values.append(er)
                    eps_im_values.append(ei)
                except Exception:
                    continue

        if not energies:
            raise ValueError("No numeric rows could be read from the selected material file.")

        self._cache = (
            np.array(energies, dtype=float),
            np.array(n_values, dtype=float),
            np.array(k_values, dtype=float),
            np.array(eps_re_values, dtype=float),
            np.array(eps_im_values, dtype=float),
        )
        return self._cache

    def to_dict(self):
        d = asdict(self)
        d.pop("_cache", None)
        d["n_value"] = [self.n_value.real, self.n_value.imag]
        d["eps_value"] = [self.eps_value.real, self.eps_value.imag]
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.mode = d.get("mode", "n")
        nv = d.get("n_value", [1.0, 0.0])
        ev = d.get("eps_value", [1.0, 0.0])
        m.n_value = complex(nv[0], nv[1]) if isinstance(nv, (list, tuple)) else complex(nv)
        m.eps_value = complex(ev[0], ev[1]) if isinstance(ev, (list, tuple)) else complex(ev)
        m.file_path = d.get("file_path", "")
        m.file_x_kind = d.get("file_x_kind", "energy_ev")
        m.file_data_kind = d.get("file_data_kind", "nk")
        m.file_x_col = int(d.get("file_x_col", 0))
        m.file_n_col = int(d.get("file_n_col", 1))
        m.file_k_col = int(d.get("file_k_col", 2))
        m.file_eps_re_col = int(d.get("file_eps_re_col", 1))
        m.file_eps_im_col = int(d.get("file_eps_im_col", 2))
        m.file_delimiter = d.get("file_delimiter", "auto")
        return m


@dataclass
class TopLayer:
    name: str = "layer"
    thickness_um: float = 0.100
    material: Material = field(default_factory=Material)
    use_in_reference: bool = True

    def epsilon(self, energy_ev: float) -> complex:
        return self.material.epsilon(energy_ev)

    def to_dict(self):
        return {
            "name": self.name,
            "thickness_um": self.thickness_um,
            "material": self.material.to_dict(),
            "use_in_reference": self.use_in_reference,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name", "layer"),
            thickness_um=float(d.get("thickness_um", 0.1)),
            material=Material.from_dict(d.get("material", {})),
            use_in_reference=bool(d.get("use_in_reference", True)),
        )


def eps_gold_drude(E_eV):
    eps_inf = 9.84
    wp = 9.03
    gamma = 0.072
    E = E_eV
    return eps_inf - wp**2 / (E * (E + 1j * gamma))


def gold_epsilon(params, E_eV):
    """Return Au epsilon. Default is the built-in Drude model; optional spectral file uses the same Material class/mapping dialog as top layers."""
    mode = params.get("gold_material_mode", "built-in Drude")
    mat = params.get("gold_material", None)
    if mode == "spectral file" and mat is not None:
        return mat.epsilon(E_eV)
    return eps_gold_drude(E_eV)


def repeat_count_from_rotation_step(angle_step_deg):
    step = abs(float(angle_step_deg)) % 180.0
    if step == 0:
        return 1
    for n in range(1, 721):
        value = (n * step) % 180.0
        if abs(value) < 1e-9 or abs(value - 180.0) < 1e-9:
            return n
    return 1


def make_macrocell_eps(
    eps_metal, period_x, period_y, slit_short, slit_long, angle_step_deg,
    n_cells_x, n_cells_y, Nx, Ny, eps_air=1.0
):
    Lx = n_cells_x * period_x
    Ly = n_cells_y * period_y
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    eps = np.ones((Nx, Ny), dtype=complex) * eps_metal

    for ix in range(n_cells_x):
        for iy in range(n_cells_y):
            cx = -Lx / 2 + (ix + 0.5) * period_x
            cy = -Ly / 2 + (iy + 0.5) * period_y
            theta = np.deg2rad(angle_step_deg * ix)
            dx = X - cx
            dy = Y - cy
            xp = dx * np.cos(theta) + dy * np.sin(theta)
            yp = -dx * np.sin(theta) + dy * np.cos(theta)
            slit = (np.abs(xp) <= slit_short / 2) & (np.abs(yp) <= slit_long / 2)
            eps[slit] = eps_air

    return eps, Lx, Ly


def load_cell_design_json(path):
    """Load a lighTTice Designer cell JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Cell file must be a JSON object.")
    if "shapes" not in data:
        raise ValueError("Cell file does not contain a 'shapes' list.")
    if "Lx" not in data or "Ly" not in data:
        # also accept a nested domain object
        dom = data.get("domain", {})
        if "Lx" in dom and "Ly" in dom:
            data["Lx"] = dom["Lx"]
            data["Ly"] = dom["Ly"]
        else:
            raise ValueError("Cell file must define Lx and Ly in microns.")
    return data


def _as_vec2(v, default):
    try:
        if isinstance(v, dict):
            return np.array([float(v.get("x", default[0])), float(v.get("y", default[1]))], dtype=float)
        return np.array([float(v[0]), float(v[1])], dtype=float)
    except Exception:
        return np.array(default, dtype=float)


def get_primitive_lattice_vectors(params):
    """Primitive 2D Bravais lattice vectors in microns.

    For ordinary/built-in geometries the vectors are rectangular:
        a1=(period_x,0), a2=(0,period_y).
    If a designer cell JSON contains a1/a2, those vectors are used.
    This enables true oblique/hexagonal cells in grcwa.
    """
    px = float(params.get("period_x", 1.0))
    py = float(params.get("period_y", 1.0))
    design = params.get("cell_design") or {}
    if design and ("a1" in design and "a2" in design):
        return _as_vec2(design.get("a1"), [px, 0.0]), _as_vec2(design.get("a2"), [0.0, py])
    lat = design.get("lattice", {}) if isinstance(design.get("lattice", {}), dict) else {}
    if lat and ("a1" in lat and "a2" in lat):
        return _as_vec2(lat.get("a1"), [px, 0.0]), _as_vec2(lat.get("a2"), [0.0, py])
    return np.array([px, 0.0], dtype=float), np.array([0.0, py], dtype=float)


def get_grcwa_lattice_vectors(params):
    """Supercell lattice vectors passed to grcwa.obj()."""
    a1, a2 = get_primitive_lattice_vectors(params)
    A1 = a1 * float(params.get("n_cells_x", 1))
    A2 = a2 * float(params.get("n_cells_y", 1))
    return A1.tolist(), A2.tolist()


def get_lattice_dims_from_params(params):
    """Return an axis-aligned bounding-box size for previews/labels only."""
    A1, A2 = [np.asarray(v, dtype=float) for v in get_grcwa_lattice_vectors(params)]
    corners = np.array([[-0.5*A1[0]-0.5*A2[0], -0.5*A1[1]-0.5*A2[1]],
                        [ 0.5*A1[0]-0.5*A2[0],  0.5*A1[1]-0.5*A2[1]],
                        [-0.5*A1[0]+0.5*A2[0], -0.5*A1[1]+0.5*A2[1]],
                        [ 0.5*A1[0]+0.5*A2[0],  0.5*A1[1]+0.5*A2[1]]])
    return float(corners[:,0].max()-corners[:,0].min()), float(corners[:,1].max()-corners[:,1].min())


def lattice_edge_mesh(params, Nx=None, Ny=None):
    """Physical corner mesh for arrays sampled on the lattice parallelogram.

    The solver grid is indexed by fractional coordinates (u,v) over the
    supercell lattice vectors A1,A2.  This helper is only for correct physical
    previews/plots; it prevents oblique cells from being displayed as if they
    were rectangular images.
    """
    if Nx is None:
        Nx = int(params.get("Nx", 160))
    if Ny is None:
        Ny = int(params.get("Ny", 160))
    A1, A2 = [np.asarray(v, dtype=float) for v in get_grcwa_lattice_vectors(params)]
    ue = np.linspace(-0.5, 0.5, int(Nx) + 1)
    ve = np.linspace(-0.5, 0.5, int(Ny) + 1)
    Ue, Ve = np.meshgrid(ue, ve, indexing="ij")
    Xe = Ue * A1[0] + Ve * A2[0]
    Ye = Ue * A1[1] + Ve * A2[1]
    return Xe, Ye


def plot_lattice_array(ax, Z, params, title=None, cmap=None, vmin=None, vmax=None):
    """Plot a [Nx,Ny] lattice-sampled array in physical x/y coordinates."""
    Z = np.asarray(Z)
    Nx, Ny = Z.shape[:2]
    Xe, Ye = lattice_edge_mesh(params, Nx, Ny)
    im = ax.pcolormesh(Xe, Ye, Z, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    if title:
        ax.set_title(title)
    return im


def _shape_mask_local(xl, yl, sh):
    """Mask for one designer shape evaluated in local unit-cell coordinates."""
    typ = str(sh.get("type", "slit")).lower()
    cx = float(sh.get("cx", sh.get("x", 0.0)))
    cy = float(sh.get("cy", sh.get("y", 0.0)))
    angle = np.deg2rad(float(sh.get("angle", sh.get("theta", 0.0))))
    dx = xl - cx
    dy = yl - cy
    xp = dx * np.cos(angle) + dy * np.sin(angle)
    yp = -dx * np.sin(angle) + dy * np.cos(angle)

    if typ in ("slit", "rectangle", "rect", "box"):
        width = float(sh.get("width", sh.get("short", sh.get("w", 0.05))))
        height = float(sh.get("height", sh.get("long", sh.get("length", sh.get("h", 0.30)))))
        return (np.abs(xp) <= width / 2) & (np.abs(yp) <= height / 2)
    if typ in ("ellipse", "elliptical"):
        rx = float(sh.get("rx", sh.get("radius_x", sh.get("width", 0.10) / 2)))
        ry = float(sh.get("ry", sh.get("radius_y", sh.get("height", 0.10) / 2)))
        return (xp / rx) ** 2 + (yp / ry) ** 2 <= 1.0
    if typ == "circle":
        r = float(sh.get("r", sh.get("radius", 0.05)))
        return xp ** 2 + yp ** 2 <= r ** 2
    return None


def make_cell_design_macrocell_eps(
    eps_metal, cell_design, period_x, period_y, angle_step_deg,
    n_cells_x, n_cells_y, Nx, Ny, eps_air=1.0
):
    """Rasterize a designer unit-cell motif into the solver supercell.

    Supports true oblique/hexagonal Bravais lattices when the cell JSON stores
    primitive vectors a1 and a2.  The grcwa grid is sampled on the supercell
    parallelogram spanned by A1=n_cells_x*a1 and A2=n_cells_y*a2.

    Shape coordinates remain ordinary Cartesian microns in the local cell frame.
    The motif is repeated by primitive translations ix*a1 + iy*a2; optional
    macrocell rotation is applied per ix column just like the original slit code.
    """
    a1, a2 = get_primitive_lattice_vectors({
        "period_x": period_x, "period_y": period_y, "cell_design": cell_design
    })
    n_cells_x = int(n_cells_x); n_cells_y = int(n_cells_y)
    Nx = int(Nx); Ny = int(Ny)
    A1 = a1 * float(n_cells_x)
    A2 = a2 * float(n_cells_y)

    # Grid points in the supercell parallelogram, centered at the origin.
    u = (np.arange(Nx) + 0.5) / Nx - 0.5
    v = (np.arange(Ny) + 0.5) / Ny - 0.5
    U, V = np.meshgrid(u, v, indexing="ij")
    X = U * A1[0] + V * A2[0]
    Y = U * A1[1] + V * A2[1]

    background = str(cell_design.get("background", "metal")).lower()
    eps = np.ones((Nx, Ny), dtype=complex) * (eps_air if background in ("air", "void") else eps_metal)

    # Shapes are defined in physical Cartesian coordinates relative to the
    # primitive-cell centre.  The epsilon grid itself is sampled in fractional
    # lattice coordinates over the supercell parallelogram.  We also draw
    # neighbouring periodic images so holes crossing the cell boundary wrap
    # correctly.  This is important for true hexagonal/oblique primitive cells.
    for ix in range(n_cells_x):
        for iy in range(n_cells_y):
            cc = ((ix + 0.5) / n_cells_x - 0.5) * A1 + ((iy + 0.5) / n_cells_y - 0.5) * A2
            theta = np.deg2rad(float(angle_step_deg) * ix)

            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    cc_img = cc + ox * a1 + oy * a2
                    dx = X - cc_img[0]
                    dy = Y - cc_img[1]
                    xl = dx * np.cos(theta) + dy * np.sin(theta)
                    yl = -dx * np.sin(theta) + dy * np.cos(theta)

                    for sh in cell_design.get("shapes", []):
                        mask = _shape_mask_local(xl, yl, sh)
                        if mask is None:
                            continue
                        mat = str(sh.get("material", "air")).lower()
                        fill = eps_metal if mat in ("metal", "gold", "au") else eps_air
                        eps[mask] = fill

    # Axis-aligned bounding box dimensions for previews only.
    corners = np.array([-0.5*A1-0.5*A2, 0.5*A1-0.5*A2, -0.5*A1+0.5*A2, 0.5*A1+0.5*A2])
    Lx = float(corners[:,0].max() - corners[:,0].min())
    Ly = float(corners[:,1].max() - corners[:,1].min())
    return eps, Lx, Ly

def make_pattern_eps(params, eps_metal, eps_air=None):
    """Return the patterned-layer eps grid from either built-in slit parameters or a loaded cell file."""
    if eps_air is None:
        eps_air = params["n_air"] ** 2
    design = params.get("cell_design")
    if design:
        return make_cell_design_macrocell_eps(
            eps_metal, design, params["period_x"], params["period_y"],
            params["rotation_step"], params["n_cells_x"], params["n_cells_y"],
            params["Nx"], params["Ny"], eps_air
        )
    return make_macrocell_eps(
        eps_metal, params["period_x"], params["period_y"],
        params["slit_short"], params["slit_long"], params["rotation_step"],
        params["n_cells_x"], params["n_cells_y"], params["Nx"], params["Ny"], eps_air
    )


def incidence_angles(E_eV, kx_um, ky_um, n_incident):
    wavelength = HC_EV_UM / E_eV
    k0 = 2 * np.pi / wavelength
    kt = np.sqrt(kx_um**2 + ky_um**2)
    arg = kt / (n_incident * k0)
    if arg >= 1:
        return None
    theta = np.arcsin(arg)
    phi = np.arctan2(ky_um, kx_um) if kt > 0 else 0.0
    freq = 1.0 / wavelength
    return freq, theta, phi


def make_excitation(obj, pol):
    """
    Excitation definitions used by the solver.

    Linear:
        p: p_amp=1, s_amp=0
        s: p_amp=0, s_amp=1

    Circular (direct simulation, no Jones reconstruction required for total R/T):
        RCP = (p + i s)/sqrt(2)
        LCP = (p - i s)/sqrt(2)

    Note: the absolute handedness label depends on the optics convention and
    propagation direction. The important point is that RCP/LCP here are a
    conjugate pair with exactly controlled +/- pi/2 phase between p and s.
    """
    pol_l = str(pol).strip().lower()
    invsqrt2 = 1.0 / np.sqrt(2.0)
    if pol_l.startswith(("rcp", "r+", "right")):
        obj.MakeExcitationPlanewave(
            p_amp=invsqrt2, p_phase=0.0,
            s_amp=invsqrt2, s_phase=np.pi/2,
            order=0,
        )
    elif pol_l.startswith(("lcp", "l-", "left")):
        obj.MakeExcitationPlanewave(
            p_amp=invsqrt2, p_phase=0.0,
            s_amp=invsqrt2, s_phase=-np.pi/2,
            order=0,
        )
    elif pol_l.startswith(("p", "tm")):
        obj.MakeExcitationPlanewave(p_amp=1.0, p_phase=0.0, s_amp=0.0, s_phase=0.0, order=0)
    else:
        obj.MakeExcitationPlanewave(p_amp=0.0, p_phase=0.0, s_amp=1.0, s_phase=0.0, order=0)


def add_top_layers(obj, params, E_eV, reference=False):
    for layer in params.get("top_layers", []):
        if reference and not layer.use_in_reference:
            continue
        if layer.thickness_um > 0:
            obj.Add_LayerUniform(layer.thickness_um, layer.epsilon(E_eV))


def solve_patterned_rcwa(params, E_eV, kx_um, ky_um):
    if grcwa is None:
        raise ImportError("grcwa is not installed. Run: pip install grcwa numpy matplotlib scipy PyQt5")

    angles = incidence_angles(E_eV, kx_um, ky_um, params["n_air"])
    if angles is None:
        return np.nan, np.nan, np.nan

    freq, theta, phi = angles
    eps_air = params["n_air"] ** 2
    eps_glass = params["n_glass"] ** 2
    eps_metal = gold_epsilon(params, E_eV)

    eps_grid, Lx, Ly = make_pattern_eps(params, eps_metal, eps_air)

    a1_vec, a2_vec = get_grcwa_lattice_vectors(params)
    obj = grcwa.obj(params["nG"], a1_vec, a2_vec, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(0.0, eps_air)
    add_top_layers(obj, params, E_eV, reference=False)
    obj.Add_LayerGrid(params["gold_thickness"], params["Nx"], params["Ny"])
    obj.Add_LayerUniform(0.0, eps_glass)
    obj.Init_Setup(Gmethod=0)
    obj.GridLayer_geteps(eps_grid.flatten())
    make_excitation(obj, params["pol"])
    R, T = obj.RT_Solve(normalize=1)
    A = 1.0 - R - T
    return float(np.real(R)), float(np.real(T)), float(np.real(A))


def solve_reference_stack(params, E_eV, kx_um, ky_um):
    """
    User-selectable reference calculation.

    params["reference_mode"]:
        - "uniform_gold": selected reference top layers + uniform Au film
        - "patterned_gold": selected reference top layers + same slitted Au pattern
        - "no_gold": selected reference top layers + glass only

    The main calculation always uses all top layers and the patterned Au.
    The reference calculation uses only top layers where use_in_reference=True.
    """
    if grcwa is None:
        raise ImportError("grcwa is not installed. Run: pip install grcwa numpy matplotlib scipy PyQt5")

    angles = incidence_angles(E_eV, kx_um, ky_um, params["n_air"])
    if angles is None:
        return np.nan, np.nan, np.nan

    freq, theta, phi = angles
    eps_air = params["n_air"] ** 2
    eps_glass = params["n_glass"] ** 2
    eps_metal = gold_epsilon(params, E_eV)
    Lx, Ly = get_lattice_dims_from_params(params)

    mode = params.get("reference_mode", "uniform_gold")

    a1_vec, a2_vec = get_grcwa_lattice_vectors(params)
    obj = grcwa.obj(params["nG"], a1_vec, a2_vec, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(0.0, eps_air)
    add_top_layers(obj, params, E_eV, reference=True)

    if mode == "uniform_gold":
        obj.Add_LayerUniform(params["gold_thickness"], eps_metal)
        obj.Add_LayerUniform(0.0, eps_glass)
        obj.Init_Setup(Gmethod=0)

    elif mode == "patterned_gold":
        eps_grid, _, _ = make_pattern_eps(params, eps_metal, eps_air)
        obj.Add_LayerGrid(params["gold_thickness"], params["Nx"], params["Ny"])
        obj.Add_LayerUniform(0.0, eps_glass)
        obj.Init_Setup(Gmethod=0)
        obj.GridLayer_geteps(eps_grid.flatten())

    elif mode == "no_gold":
        # Bare selected stack on glass. The same supercell is used so k-space sampling is consistent.
        obj.Add_LayerUniform(0.0, eps_glass)
        obj.Init_Setup(Gmethod=0)

    else:
        raise ValueError(f"Unknown reference_mode: {mode}")

    make_excitation(obj, params["pol"])
    R, T = obj.RT_Solve(normalize=1)
    A = 1.0 - R - T
    return float(np.real(R)), float(np.real(T)), float(np.real(A))

def get_z_poynting_flux_np(ai, bi, omega, kp, phi, q, byorder=0):
    """Numpy copy of grcwa.rcwa.GetZPoyntingFlux.

    It converts complex modal amplitudes into z-directed Poynting flux.
    With byorder=1 it returns one value per Fourier/diffraction order.
    """
    ai = np.asarray(ai, dtype=complex)
    bi = np.asarray(bi, dtype=complex)
    kp = np.asarray(kp, dtype=complex)
    phi = np.asarray(phi, dtype=complex)
    q = np.asarray(q, dtype=complex)
    n2 = len(ai)
    n = n2 // 2

    Aop = kp @ phi @ np.diag(1.0 / (omega * q))
    pa = phi @ ai
    pb = phi @ bi
    Aa = Aop @ ai
    Ab = Aop @ bi

    diff = 0.5 * (np.conj(pb) * Aa - np.conj(Ab) * pa)
    forward_xy = np.real(np.conj(Aa) * pa) + diff
    backward_xy = -np.real(np.conj(Ab) * pb) + np.conj(diff)

    forward = forward_xy[:n] + forward_xy[n:]
    backward = backward_xy[:n] + backward_xy[n:]
    if byorder == 0:
        forward = np.sum(forward)
        backward = np.sum(backward)
    return forward, backward


def reflected_specular_from_b0(obj, b0, pol_label="p"):
    """Return specular reflected power from b0 and phase of the matching output component.

    The power is computed with the same Poynting-flux expression as RT_Solve,
    but with byorder=1 and only the G=(0,0) reflected order selected.

    The phase is extracted from the specular reflected electric field projected
    onto the same polarization family as the input: p, s, RCP, or LCP.
    """
    if b0 is None:
        return (np.nan, np.nan, np.nan + 1j*np.nan, np.nan + 1j*np.nan, np.nan + 1j*np.nan,
                np.nan + 1j*np.nan, np.nan + 1j*np.nan)

    G = np.asarray(obj.G)
    hit = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
    if len(hit) == 0:
        return (np.nan, np.nan, np.nan + 1j*np.nan, np.nan + 1j*np.nan, np.nan + 1j*np.nan,
                np.nan + 1j*np.nan, np.nan + 1j*np.nan)
    m0 = int(hit[0])

    nG = obj.nG
    b0 = np.asarray(b0, dtype=complex)
    b_spec = np.zeros_like(b0, dtype=complex)
    b_spec[m0] = b0[m0]
    b_spec[m0 + nG] = b0[m0 + nG]

    q = np.asarray(obj.q_list[0], dtype=complex)
    phi = np.asarray(obj.phi_list[0], dtype=complex)
    kp = np.asarray(obj.kp_list[0], dtype=complex)

    # Poynting flux per reflected order. Reflected flux is negative z, so R=-backward.
    _, backward_orders = get_z_poynting_flux_np(
        np.asarray(obj.a0, dtype=complex), b_spec, obj.omega, kp, phi, q, byorder=1
    )
    R_spec_b = float(np.real(-backward_orders[m0] * obj.normalization))

    # Reflected electric-field Fourier coefficient for the specular order.
    tmp = -b_spec / (obj.omega * q)
    fexy = kp @ phi @ tmp
    Ey_G = -fexy[:nG]
    Ex_G = fexy[nG:]

    Hxy = phi @ b_spec
    Hx_G = Hxy[:nG]
    Hy_G = Hxy[nG:]
    eps_top = obj.Uniform_ep_list[0]
    Ez_G = (obj.ky * Hx_G - obj.kx * Hy_G) / obj.omega / eps_top

    Ex0 = Ex_G[m0]
    Ey0 = Ey_G[m0]
    Ez0 = Ez_G[m0]
    Evec = np.array([Ex0, Ey0, Ez0], dtype=complex)

    # Build reflected s/p basis for this diffraction order.
    kx0 = float(np.real(obj.kx[m0]))
    ky0 = float(np.real(obj.ky[m0]))
    q0 = q[m0]
    kt = (kx0**2 + ky0**2)**0.5
    if kt < 1e-14:
        s_hat = np.array([0.0, 1.0, 0.0])
    else:
        s_hat = np.array([-ky0/kt, kx0/kt, 0.0])

    # Reflected wave travels toward -z.
    kvec = np.array([kx0, ky0, -float(np.real(q0))], dtype=float)
    nk = np.linalg.norm(kvec)
    khat = kvec / nk if nk > 0 else np.array([0.0, 0.0, -1.0])
    p_hat = np.cross(khat, s_hat)
    np_norm = np.linalg.norm(p_hat)
    if np_norm > 0:
        p_hat = p_hat / np_norm
    else:
        p_hat = np.array([1.0, 0.0, 0.0])

    Es = np.dot(Evec, s_hat)
    Ep = np.dot(Evec, p_hat)
    pol = str(pol_label).lower()
    if pol.startswith('s'):
        scalar = Es
    elif pol.startswith('rcp') or pol.startswith('right'):
        scalar = (Ep + 1j * Es) / np.sqrt(2.0)
    elif pol.startswith('lcp') or pol.startswith('left'):
        scalar = (Ep - 1j * Es) / np.sqrt(2.0)
    else:
        scalar = Ep
    phase = float(np.angle(scalar)) if np.isfinite(np.abs(scalar)) else np.nan

    # Circular components of the specular reflected electric field.
    # Convention is consistent with the excitation convention used above:
    # RCP = (p + i s)/sqrt(2), LCP = (p - i s)/sqrt(2).
    E_RCP_spec = (Ep + 1j * Es) / np.sqrt(2.0)
    E_LCP_spec = (Ep - 1j * Es) / np.sqrt(2.0)

    return R_spec_b, phase, Ex0, Ey0, Ez0, E_RCP_spec, E_LCP_spec

def solve_patterned_rcwa_raw(params, E_eV, kx_um, ky_um, pol):
    """
    Patterned calculation returning total R/T/A, raw reflected amplitudes b0, and raw transmitted amplitudes aN.
    """
    if grcwa is None:
        raise ImportError("grcwa is not installed. Run: pip install grcwa numpy matplotlib scipy PyQt5")

    angles = incidence_angles(E_eV, kx_um, ky_um, params["n_air"])
    if angles is None:
        return np.nan, np.nan, np.nan, None, None

    freq, theta, phi = angles
    eps_air = params["n_air"] ** 2
    eps_glass = params["n_glass"] ** 2
    eps_metal = gold_epsilon(params, E_eV)

    eps_grid, Lx, Ly = make_pattern_eps(params, eps_metal, eps_air)

    a1_vec, a2_vec = get_grcwa_lattice_vectors(params)
    obj = grcwa.obj(params["nG"], a1_vec, a2_vec, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(0.0, eps_air)
    add_top_layers(obj, params, E_eV, reference=False)
    obj.Add_LayerGrid(params["gold_thickness"], params["Nx"], params["Ny"])
    obj.Add_LayerUniform(0.0, eps_glass)
    obj.Init_Setup(Gmethod=0)
    obj.GridLayer_geteps(eps_grid.flatten())
    make_excitation(obj, pol)

    R, T = obj.RT_Solve(normalize=1)
    A = 1.0 - R - T

    try:
        ai0, b0 = obj.GetAmplitudes(0, 0.0)
        b0 = np.asarray(b0, dtype=complex)
    except Exception:
        b0 = None

    try:
        aN, _bN = obj.GetAmplitudes(len(obj.thickness_list) - 1, 0.0)
        aN = np.asarray(aN, dtype=complex)
    except Exception:
        aN = None

    try:
        G = np.asarray(obj.G)
    except Exception:
        G = None

    try:
        R_b_spec, phase_b_spec, Ex_b_spec, Ey_b_spec, Ez_b_spec, E_RCP_b_spec, E_LCP_b_spec = reflected_specular_from_b0(obj, b0, pol)
    except Exception:
        R_b_spec, phase_b_spec = np.nan, np.nan
        Ex_b_spec = Ey_b_spec = Ez_b_spec = np.nan + 1j*np.nan
        E_RCP_b_spec = E_LCP_b_spec = np.nan + 1j*np.nan

    # Save Fourier-domain fields at the top and bottom exterior layers.
    # Top layer 0 contains incident + reflected fields; the viewer subtracts
    # the known incident zeroth-order field before CP postprocessing.
    E_top, H_top = solve_field_fourier_safe(obj, 0, 0.0)
    E_bottom, H_bottom = solve_field_fourier_safe(obj, len(obj.thickness_list) - 1, 0.0)

    return (
        float(np.real(R)), float(np.real(T)), float(np.real(A)),
        b0, aN, G, E_top, H_top, E_bottom, H_bottom,
        R_b_spec, phase_b_spec, Ex_b_spec, Ey_b_spec, Ez_b_spec, E_RCP_b_spec, E_LCP_b_spec
    )


def solve_reference_rcwa_raw(params, E_eV, kx_um, ky_um, pol):
    """
    Reference calculation returning total R/T/A, raw reflected amplitudes b0, and raw transmitted amplitudes aN.
    """
    if grcwa is None:
        raise ImportError("grcwa is not installed. Run: pip install grcwa numpy matplotlib scipy PyQt5")

    angles = incidence_angles(E_eV, kx_um, ky_um, params["n_air"])
    if angles is None:
        return np.nan, np.nan, np.nan, None, None

    freq, theta, phi = angles
    eps_air = params["n_air"] ** 2
    eps_glass = params["n_glass"] ** 2
    eps_metal = gold_epsilon(params, E_eV)
    Lx, Ly = get_lattice_dims_from_params(params)

    mode = params.get("reference_mode", "uniform_gold")
    a1_vec, a2_vec = get_grcwa_lattice_vectors(params)
    obj = grcwa.obj(params["nG"], a1_vec, a2_vec, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(0.0, eps_air)
    add_top_layers(obj, params, E_eV, reference=True)

    if mode == "uniform_gold":
        obj.Add_LayerUniform(params["gold_thickness"], eps_metal)
        obj.Add_LayerUniform(0.0, eps_glass)
        obj.Init_Setup(Gmethod=0)
    elif mode == "patterned_gold":
        eps_grid, _, _ = make_pattern_eps(params, eps_metal, eps_air)
        obj.Add_LayerGrid(params["gold_thickness"], params["Nx"], params["Ny"])
        obj.Add_LayerUniform(0.0, eps_glass)
        obj.Init_Setup(Gmethod=0)
        obj.GridLayer_geteps(eps_grid.flatten())
    elif mode == "no_gold":
        obj.Add_LayerUniform(0.0, eps_glass)
        obj.Init_Setup(Gmethod=0)
    else:
        raise ValueError(f"Unknown reference_mode: {mode}")

    make_excitation(obj, pol)
    R, T = obj.RT_Solve(normalize=1)
    A = 1.0 - R - T

    try:
        ai0, b0 = obj.GetAmplitudes(0, 0.0)
        b0 = np.asarray(b0, dtype=complex)
    except Exception:
        b0 = None

    try:
        aN, _bN = obj.GetAmplitudes(len(obj.thickness_list) - 1, 0.0)
        aN = np.asarray(aN, dtype=complex)
    except Exception:
        aN = None

    try:
        G = np.asarray(obj.G)
    except Exception:
        G = None

    try:
        R_b_spec, phase_b_spec, Ex_b_spec, Ey_b_spec, Ez_b_spec, E_RCP_b_spec, E_LCP_b_spec = reflected_specular_from_b0(obj, b0, pol)
    except Exception:
        R_b_spec, phase_b_spec = np.nan, np.nan
        Ex_b_spec = Ey_b_spec = Ez_b_spec = np.nan + 1j*np.nan
        E_RCP_b_spec = E_LCP_b_spec = np.nan + 1j*np.nan

    E_top, H_top = solve_field_fourier_safe(obj, 0, 0.0)
    E_bottom, H_bottom = solve_field_fourier_safe(obj, len(obj.thickness_list) - 1, 0.0)

    return (
        float(np.real(R)), float(np.real(T)), float(np.real(A)),
        b0, aN, G, E_top, H_top, E_bottom, H_bottom,
        R_b_spec, phase_b_spec, Ex_b_spec, Ey_b_spec, Ez_b_spec, E_RCP_b_spec, E_LCP_b_spec
    )


def pol_list_from_mode(pol_mode):
    """Linear-basis solver only. Default/recommended mode runs both p and s.

    The viewer can synthesize RCP/LCP or any other incident polarization
    coherently from the saved complex b0_p and b0_s responses.
    """
    pol_mode = str(pol_mode).strip().lower()
    if pol_mode in ["p and s", "p+s", "ps", "both", "linear", "linear p+s"]:
        return ["p", "s"]
    if pol_mode.startswith("s"):
        return ["s"]
    return ["p"]

def solve_field_fourier_safe(obj, which_layer, z_offset):
    """
    Return E,H Fourier fields as arrays with shape [3, nG], or (None, None).

    grcwa docs define:
        E,H = obj.Solve_FieldFourier(which_layer, z_offset)
        E = [Ex,Ey,Ez], H = [Hx,Hy,Hz]
    """
    try:
        E, H = obj.Solve_FieldFourier(which_layer, z_offset)
        E = np.asarray(E, dtype=complex)
        H = np.asarray(H, dtype=complex)
        return E, H
    except Exception:
        return None, None


def unpack_raw_result(result):
    """Accept old/new raw solver returns and pad missing b/a-derived quantities."""
    nan_c = np.nan + 1j*np.nan
    # New v4.3 format:
    # R,T,A,b0,aN,G,E_top,H_top,E_bottom,H_bottom,R_b_spec,phase,Ex,Ey,Ez,E_RCP,E_LCP
    if len(result) == 17:
        return result
    # v4.2 format without aN
    if len(result) == 16:
        R,T,A,b0,G,E_top,H_top,E_bottom,H_bottom,R_b_spec,phase,Ex,Ey,Ez,E_RCP,E_LCP = result
        return R,T,A,b0,None,G,E_top,H_top,E_bottom,H_bottom,R_b_spec,phase,Ex,Ey,Ez,E_RCP,E_LCP
    if len(result) == 14:
        R,T,A,b0,G,E_top,H_top,E_bottom,H_bottom,R_b_spec,phase,Ex,Ey,Ez = result
        return R,T,A,b0,None,G,E_top,H_top,E_bottom,H_bottom,R_b_spec,phase,Ex,Ey,Ez,nan_c,nan_c
    if len(result) == 9:
        R, T, A, b0, G, E_top, H_top, E_bottom, H_bottom = result
        return R, T, A, b0, None, G, E_top, H_top, E_bottom, H_bottom, np.nan, np.nan, nan_c, nan_c, nan_c, nan_c, nan_c
    if len(result) == 5:
        R, T, A, b0, G = result
        return R, T, A, b0, None, G, None, None, None, None, np.nan, np.nan, nan_c, nan_c, nan_c, nan_c, nan_c
    raise ValueError(f"Unexpected raw solver return length: {len(result)}")


def solve_field_on_grid_compat(obj, layer_index, z_offset, Nxy):
    """Compatibility wrapper for different grcwa versions.

    Some grcwa releases expose Solve_FieldOnGrid(which_layer, z_offset, Nxy=None),
    while others expose only Solve_FieldOnGrid(which_layer, z_offset).  If the
    installed method cannot accept Nxy, reconstruct the real-space grid manually
    from Solve_FieldFourier using grcwa.get_ifft.
    """
    # Newer grcwa API: allows an explicit grid size.
    try:
        return obj.Solve_FieldOnGrid(layer_index, z_offset, Nxy=Nxy)
    except TypeError:
        pass
    try:
        return obj.Solve_FieldOnGrid(layer_index, z_offset, Nxy)
    except TypeError:
        pass

    # Older grcwa API: manual inverse Fourier reconstruction.
    if grcwa is None or not hasattr(grcwa, "get_ifft"):
        raise RuntimeError("This grcwa version does not accept Nxy and grcwa.get_ifft is unavailable.")

    Nx, Ny = int(Nxy[0]), int(Nxy[1])
    feh = obj.Solve_FieldFourier(layer_index, z_offset)

    # In grcwa versions based on this codebase, scalar z_offset returns a list
    # with one element: [[[fEx,fEy,fEz],[fHx,fHy,fHz]]].  Be defensive.
    if isinstance(feh, (list, tuple)) and len(feh) == 1:
        feh0 = feh[0]
    else:
        feh0 = feh

    fe = feh0[0]
    fh = feh0[1]
    Ex = grcwa.get_ifft(Nx, Ny, fe[0], obj.G)
    Ey = grcwa.get_ifft(Nx, Ny, fe[1], obj.G)
    Ez = grcwa.get_ifft(Nx, Ny, fe[2], obj.G)
    Hx = grcwa.get_ifft(Nx, Ny, fh[0], obj.G)
    Hy = grcwa.get_ifft(Nx, Ny, fh[1], obj.G)
    Hz = grcwa.get_ifft(Nx, Ny, fh[2], obj.G)
    return [[Ex, Ey, Ez], [Hx, Hy, Hz]]

def solve_patterned_field_on_grid(params, E_eV, kx_um, ky_um, pol, which="gold", z_fraction=0.5, Nxy=None):
    """
    Single-point real-space complex field preview for the patterned stack.

    Returns a dict with complex Ex/Ey/Ez/Hx/Hy/Hz on the requested real-space grid.
    This is intentionally single-energy/single-k because full scans would create huge files.
    """
    if grcwa is None:
        raise ImportError("grcwa is not installed. Run: pip install grcwa numpy matplotlib scipy PyQt5")

    angles = incidence_angles(E_eV, kx_um, ky_um, params["n_air"])
    if angles is None:
        raise ValueError("Requested k-point is outside the incident medium light cone for this energy.")

    freq, theta, phi = angles
    eps_air = params["n_air"] ** 2
    eps_glass = params["n_glass"] ** 2
    eps_metal = gold_epsilon(params, E_eV)

    eps_grid, Lx, Ly = make_pattern_eps(params, eps_metal, eps_air)

    a1_vec, a2_vec = get_grcwa_lattice_vectors(params)
    obj = grcwa.obj(params["nG"], a1_vec, a2_vec, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(0.0, eps_air)
    add_top_layers(obj, params, E_eV, reference=False)
    gold_layer_index = len(obj.thickness_list)
    obj.Add_LayerGrid(params["gold_thickness"], params["Nx"], params["Ny"])
    glass_layer_index = len(obj.thickness_list)
    obj.Add_LayerUniform(0.0, eps_glass)
    obj.Init_Setup(Gmethod=0)
    obj.GridLayer_geteps(eps_grid.flatten())
    make_excitation(obj, pol)

    R, T = obj.RT_Solve(normalize=1)
    A = 1.0 - R - T

    which_norm = str(which).lower().strip()
    if which_norm.startswith("top") or which_norm.startswith("air"):
        layer_index = 0
        thickness = 0.0
        z_offset = 0.0
    elif which_norm.startswith("bottom") or which_norm.startswith("glass"):
        layer_index = glass_layer_index
        thickness = 0.0
        z_offset = 0.0
    else:
        layer_index = gold_layer_index
        thickness = float(params["gold_thickness"])
        z_offset = float(np.clip(z_fraction, 0.0, 1.0)) * thickness

    if Nxy is None:
        Nxy = [int(params["Nx"]), int(params["Ny"])]

    Efield, Hfield = solve_field_on_grid_compat(obj, layer_index, z_offset, Nxy)
    Efield = [np.asarray(c, dtype=complex) for c in Efield]
    Hfield = [np.asarray(c, dtype=complex) for c in Hfield]

    Ex, Ey, Ez = Efield
    Hx, Hy, Hz = Hfield
    intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    return {
        "E_eV": float(E_eV), "kx": float(kx_um), "ky": float(ky_um), "pol": str(pol),
        "R": float(np.real(R)), "T": float(np.real(T)), "A": float(np.real(A)),
        "which_layer": which, "layer_index": int(layer_index), "z_offset_um": float(z_offset),
        "Lx": float(Lx), "Ly": float(Ly), "eps_grid": eps_grid,
        "Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz,
        "Eabs2": intensity,
    }




class MaterialFileMapperDialog(QDialog):
    def __init__(self, parent, file_path: str, material: Optional[Material] = None):
        super().__init__(parent)
        self.setWindowTitle("Map material file columns")
        self.resize(980, 640)
        self.file_path = file_path
        self.material = material or Material(mode="file", file_path=file_path)
        self.result = None

        main = QVBoxLayout(self)
        main.addWidget(QLabel(f"File: {file_path}"))

        controls_box = QGroupBox("Column mapping")
        controls = QFormLayout(controls_box)
        main.addWidget(controls_box)

        row1 = QHBoxLayout()
        self.delim_box = QComboBox(); self.delim_box.addItems(["auto", "comma", "tab", "space"])
        self.delim_box.setCurrentText(self.material.file_delimiter)
        self.data_kind_box = QComboBox(); self.data_kind_box.addItems(["nk", "epsilon"])
        self.data_kind_box.setCurrentText(self.material.file_data_kind)
        self.x_kind_box = QComboBox(); self.x_kind_box.addItems(["energy_ev", "wavelength_nm", "wavelength_um", "wavelength_m"])
        self.x_kind_box.setCurrentText(self.material.file_x_kind)
        row1.addWidget(QLabel("Delimiter")); row1.addWidget(self.delim_box)
        row1.addWidget(QLabel("Data are")); row1.addWidget(self.data_kind_box)
        row1.addWidget(QLabel("X means")); row1.addWidget(self.x_kind_box)
        controls.addRow(row1)

        row2 = QHBoxLayout()
        self.x_col = QLineEdit(str(self.material.file_x_col))
        self.n_col = QLineEdit(str(self.material.file_n_col))
        self.k_col = QLineEdit(str(self.material.file_k_col))
        self.eps_re_col = QLineEdit(str(self.material.file_eps_re_col))
        self.eps_im_col = QLineEdit(str(self.material.file_eps_im_col))
        for w in [self.x_col, self.n_col, self.k_col, self.eps_re_col, self.eps_im_col]:
            w.setMaximumWidth(60)
        row2.addWidget(QLabel("X col")); row2.addWidget(self.x_col)
        row2.addWidget(QLabel("n col")); row2.addWidget(self.n_col)
        row2.addWidget(QLabel("k col")); row2.addWidget(self.k_col)
        row2.addWidget(QLabel("eps real col")); row2.addWidget(self.eps_re_col)
        row2.addWidget(QLabel("eps imag col")); row2.addWidget(self.eps_im_col)
        controls.addRow(row2)

        row3 = QHBoxLayout()
        self.plot_view_box = QComboBox(); self.plot_view_box.addItems(["nk", "epsilon"])
        refresh_btn = QPushButton("Refresh preview"); refresh_btn.clicked.connect(self.refresh_preview)
        plot_btn = QPushButton("Plot mapped data"); plot_btn.clicked.connect(self.plot_mapped_data)
        use_btn = QPushButton("Use this mapping"); use_btn.clicked.connect(self.accept_mapping)
        cancel_btn = QPushButton("Cancel"); cancel_btn.clicked.connect(self.reject)
        row3.addWidget(QLabel("Plot view")); row3.addWidget(self.plot_view_box)
        row3.addWidget(refresh_btn); row3.addWidget(plot_btn); row3.addWidget(use_btn); row3.addWidget(cancel_btn)
        controls.addRow(row3)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        main.addWidget(self.table)

        self.status = QLabel("")
        main.addWidget(self.status)

        for box in [self.delim_box, self.data_kind_box, self.x_kind_box]:
            box.currentTextChanged.connect(self.refresh_preview)

        self.refresh_preview()

    def split_line(self, line):
        line = line.strip()
        if not line:
            return []
        d = self.delim_box.currentText()
        if d == "comma":
            return [x.strip() for x in line.split(",")]
        if d == "tab":
            return [x.strip() for x in line.split("\t")]
        if d == "space":
            return line.split()
        return [x.strip() for x in line.split(",")] if "," in line else line.split()

    def read_preview_rows(self, max_rows=40):
        rows = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = self.split_line(line)
                if parts:
                    rows.append(parts)
                if len(rows) >= max_rows:
                    break
        return rows

    def material_from_controls(self):
        return Material(
            mode="file",
            file_path=self.file_path,
            file_delimiter=self.delim_box.currentText(),
            file_x_kind=self.x_kind_box.currentText(),
            file_data_kind=self.data_kind_box.currentText(),
            file_x_col=int(self.x_col.text()),
            file_n_col=int(self.n_col.text()),
            file_k_col=int(self.k_col.text()),
            file_eps_re_col=int(self.eps_re_col.text()),
            file_eps_im_col=int(self.eps_im_col.text()),
        )

    def refresh_preview(self):
        try:
            rows = self.read_preview_rows()
            ncols = max((len(r) for r in rows), default=0)
            self.table.clear()
            self.table.setRowCount(min(25, len(rows)))
            self.table.setColumnCount(ncols)
            self.table.setHorizontalHeaderLabels([f"col {i}" for i in range(ncols)])
            for i, row in enumerate(rows[:25]):
                for j in range(ncols):
                    self.table.setItem(i, j, QTableWidgetItem(row[j] if j < len(row) else ""))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.status.setText(self.try_mapped_preview(rows))
        except Exception as e:
            self.status.setText(f"Preview error: {e}")

    def try_mapped_preview(self, rows):
        try:
            mat = self.material_from_controls()
            xcol = mat.file_x_col
            if mat.file_data_kind == "epsilon":
                c1, c2 = mat.file_eps_re_col, mat.file_eps_im_col
                label = "eps real, eps imag"
            else:
                c1, c2 = mat.file_n_col, mat.file_k_col
                label = "n, k"
            out = []
            for r in rows:
                try:
                    x = float(r[xcol]); a = float(r[c1]); b = float(r[c2])
                except Exception:
                    continue
                if mat.file_x_kind == "energy_ev":
                    E = x
                elif mat.file_x_kind == "wavelength_nm":
                    E = HC_EV_NM / x
                elif mat.file_x_kind == "wavelength_um":
                    E = HC_EV_UM / x
                else:
                    E = HC_EV_UM / (x * 1e6)
                out.append((E, a, b))
                if len(out) >= 3:
                    break
            if not out:
                return "No numeric mapped rows found. Adjust delimiter/units/columns."
            return f"Mapped sample ({label}): " + "; ".join([f"E={E:.4g} eV, a={a:.4g}, b={b:.4g}" for E, a, b in out])
        except Exception as e:
            return f"Mapping not valid yet: {e}"

    def plot_mapped_data(self):
        try:
            mat = self.material_from_controls()
            E, n, k, er, ei = mat.load_arrays()
            fig = Figure(figsize=(7, 4.5), dpi=100)
            ax = fig.add_subplot(111)

            if self.plot_view_box.currentText() == "epsilon":
                ax.plot(E, er, label="eps real")
                ax.plot(E, ei, label="eps imag")
                ax.set_ylabel("epsilon")
            else:
                ax.plot(E, n, label="n")
                ax.plot(E, k, label="k")
                ax.set_ylabel("n, k")
            ax.set_xlabel("Energy (eV)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Mapped material file")

            dlg = QDialog(self)
            dlg.setWindowTitle("Material file plot")
            lay = QVBoxLayout(dlg)
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, dlg)
            lay.addWidget(toolbar)
            lay.addWidget(canvas)
            dlg.resize(760, 520)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Plot error", str(e))

    def accept_mapping(self):
        try:
            mat = self.material_from_controls()
            mat.load_arrays()
            self.result = mat
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Invalid mapping", str(e))


class LayerDialog(QDialog):
    def __init__(self, parent, layer: Optional[TopLayer] = None):
        super().__init__(parent)
        self.setWindowTitle("Homogeneous top layer")
        self.resize(680, 380)
        self.result = None
        self.layer = layer or TopLayer()

        main = QVBoxLayout(self)
        form = QFormLayout()
        main.addLayout(form)

        self.name_edit = QLineEdit(self.layer.name)
        self.thick_edit = QLineEdit(str(self.layer.thickness_um))
        self.ref_check = QCheckBox("Use this layer in the reference stack")
        self.ref_check.setChecked(self.layer.use_in_reference)
        form.addRow("Name", self.name_edit)
        form.addRow("Thickness, um", self.thick_edit)
        form.addRow("Reference", self.ref_check)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["n", "eps", "file"])
        self.mode_box.setCurrentText(self.layer.material.mode)
        form.addRow("Material input", self.mode_box)

        nrow = QHBoxLayout()
        self.n_re = QLineEdit(str(self.layer.material.n_value.real))
        self.n_im = QLineEdit(str(self.layer.material.n_value.imag))
        nrow.addWidget(QLabel("real")); nrow.addWidget(self.n_re)
        nrow.addWidget(QLabel("imag/k")); nrow.addWidget(self.n_im)
        form.addRow("constant n", nrow)

        erow = QHBoxLayout()
        self.eps_re = QLineEdit(str(self.layer.material.eps_value.real))
        self.eps_im = QLineEdit(str(self.layer.material.eps_value.imag))
        erow.addWidget(QLabel("real")); erow.addWidget(self.eps_re)
        erow.addWidget(QLabel("imag")); erow.addWidget(self.eps_im)
        form.addRow("constant epsilon", erow)

        filerow = QHBoxLayout()
        self.file_edit = QLineEdit(self.layer.material.file_path)
        browse_btn = QPushButton("Browse / map")
        browse_btn.clicked.connect(self.browse_map)
        remap_btn = QPushButton("View / plot / remap")
        remap_btn.clicked.connect(self.remap)
        filerow.addWidget(self.file_edit)
        filerow.addWidget(browse_btn)
        filerow.addWidget(remap_btn)
        form.addRow("spectral file", filerow)

        self.file_material = Material.from_dict(self.layer.material.to_dict())
        self.file_material.mode = "file"

        info = QLabel(
            "File mapping is saved inside the layer. Use Browse/map to preview the file, map columns, and plot n,k or epsilon."
        )
        info.setWordWrap(True)
        main.addWidget(info)

        buttons = QHBoxLayout()
        ok = QPushButton("OK"); ok.clicked.connect(self.ok)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        buttons.addStretch(1); buttons.addWidget(ok); buttons.addWidget(cancel)
        main.addLayout(buttons)

    def browse_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select material file", "", "Data files (*.csv *.txt *.dat);;All files (*)"
        )
        if not path:
            return
        self.file_edit.setText(path)
        self.file_material.file_path = path
        dlg = MaterialFileMapperDialog(self, path, self.file_material)
        if dlg.exec_() == QDialog.Accepted and dlg.result is not None:
            self.file_material = dlg.result
            self.file_edit.setText(self.file_material.file_path)
            self.mode_box.setCurrentText("file")

    def remap(self):
        path = self.file_edit.text().strip()
        if not path:
            QMessageBox.information(self, "No file", "No material file selected yet.")
            return
        self.file_material.file_path = path
        dlg = MaterialFileMapperDialog(self, path, self.file_material)
        if dlg.exec_() == QDialog.Accepted and dlg.result is not None:
            self.file_material = dlg.result
            self.file_edit.setText(self.file_material.file_path)
            self.mode_box.setCurrentText("file")

    def ok(self):
        try:
            mode = self.mode_box.currentText()
            if mode == "file":
                material = self.file_material
                material.mode = "file"
                material.file_path = self.file_edit.text().strip()
                material.load_arrays()
            elif mode == "eps":
                material = Material(
                    mode="eps",
                    eps_value=complex(float(self.eps_re.text()), float(self.eps_im.text()))
                )
            else:
                material = Material(
                    mode="n",
                    n_value=complex(float(self.n_re.text()), float(self.n_im.text()))
                )

            self.result = TopLayer(
                name=self.name_edit.text().strip() or "layer",
                thickness_um=float(self.thick_edit.text()),
                material=material,
                use_in_reference=self.ref_check.isChecked(),
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Invalid layer", str(e))


def _set_single_thread_blas_env():
    """Avoid oversubscription when several RCWA workers run at once."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


# ---------- Precomputed common polarization maps saved for faster viewer startup ----------

def _safe_divide_num(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    np.divide(a, b, out=out, where=np.isfinite(b) & (np.abs(b) > 1e-15))
    return out


def _make_kp_matrix_uniform_np(omega, epsilon, kx, ky):
    """Numpy copy of grcwa's uniform-layer kp matrix."""
    kx = np.asarray(kx, dtype=complex)
    ky = np.asarray(ky, dtype=complex)
    nG = len(kx)
    Jk = np.vstack((-np.diag(ky), np.diag(kx)))
    JkkJT = Jk @ Jk.T
    return omega**2 * np.eye(2*nG, dtype=complex) - (1.0/epsilon) * JkkJT


def _get_z_poynting_flux_np(ai, bi, omega, kp, phi, q, byorder=False):
    """Numpy copy of grcwa GetZPoyntingFlux for exterior uniform layers."""
    ai = np.asarray(ai, dtype=complex)
    bi = np.asarray(bi, dtype=complex)
    q = np.asarray(q, dtype=complex)
    phi = np.asarray(phi, dtype=complex)
    kp = np.asarray(kp, dtype=complex)
    n2 = len(ai)
    n = n2 // 2
    A = kp @ phi @ np.diag(1.0 / (omega * q))
    pa = phi @ ai
    pb = phi @ bi
    Aa = A @ ai
    Ab = A @ bi
    diff = 0.5 * (np.conj(pb) * Aa - np.conj(Ab) * pa)
    forward_xy = np.real(np.conj(Aa) * pa) + diff
    backward_xy = -np.real(np.conj(Ab) * pb) + np.conj(diff)
    forward = forward_xy[:n] + forward_xy[n:]
    backward = backward_xy[:n] + backward_xy[n:]
    if not byorder:
        forward = np.sum(forward)
        backward = np.sum(backward)
    return forward, backward


def _basis_vector_sp(label):
    """Return [s,p] amplitudes. Convention RCP=(p+i*s)/sqrt(2)."""
    lab = str(label).lower().strip()
    if lab.startswith('s'):
        return np.array([1.0, 0.0], dtype=complex)
    if lab.startswith('p'):
        return np.array([0.0, 1.0], dtype=complex)
    if lab.startswith('rcp'):
        return np.array([1j, 1.0], dtype=complex) / np.sqrt(2.0)
    if lab.startswith('lcp'):
        return np.array([-1j, 1.0], dtype=complex) / np.sqrt(2.0)
    raise ValueError(f'Unknown polarization {label}')


def _zero_order_index_from_G(G_orders):
    try:
        G = np.asarray(G_orders)
        idx = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
        return int(idx[0]) if len(idx) else 0
    except Exception:
        return 0


def _incident_a0_vector_np(incident, E_eV, kx_inc, ky_inc, nG, G_orders, n_air):
    vin = _basis_vector_sp(incident)
    s_amp, p_amp = complex(vin[0]), complex(vin[1])
    a0 = np.zeros(2*nG, dtype=complex)
    m0 = _zero_order_index_from_G(G_orders)
    k0 = 2*np.pi*float(E_eV)/HC_EV_UM
    nk0 = float(n_air) * k0
    kt = np.hypot(float(kx_inc), float(ky_inc))
    theta = np.arcsin(np.clip(kt/max(nk0, 1e-30), -1.0, 1.0))
    phi_ang = np.arctan2(float(ky_inc), float(kx_inc)) if kt > 1e-15 else 0.0
    a0[m0] = (-s_amp*np.cos(theta)*np.cos(phi_ang) - p_amp*np.sin(phi_ang))
    a0[m0+nG] = (-s_amp*np.cos(theta)*np.sin(phi_ang) + p_amp*np.cos(phi_ang))
    return a0


def _order_geometry_np(params, G_orders, E_eV, kx_inc, ky_inc, side='reflection'):
    G = np.asarray(G_orders, dtype=float)
    Lx = float(params.get('n_cells_x', 1)) * float(params.get('period_x', 1.0))
    Ly = float(params.get('n_cells_y', 1)) * float(params.get('period_y', 1.0))
    kx = float(kx_inc) + 2*np.pi*G[:, 0]/Lx
    ky = float(ky_inc) + 2*np.pi*G[:, 1]/Ly
    k0 = 2*np.pi*float(E_eV)/HC_EV_UM
    nmed = float(params.get('n_air', 1.0)) if side == 'reflection' else float(params.get('n_glass', 1.5))
    nk0 = nmed * k0
    kpar2 = kx*kx + ky*ky
    prop = kpar2 <= nk0*nk0 + 1e-12
    kz_abs = np.zeros_like(kx, dtype=float)
    kz_abs[prop] = np.sqrt(np.maximum(nk0*nk0 - kpar2[prop], 0.0))
    kz = -kz_abs if side == 'reflection' else kz_abs
    return kx, ky, kz, kz_abs, prop, nk0


def _synthesize_modal_for_incident(modal, pol_labels, incident):
    labels = [str(x).lower().strip() for x in pol_labels]
    if 's' not in labels or 'p' not in labels:
        return None
    is_idx, ip_idx = labels.index('s'), labels.index('p')
    vin = _basis_vector_sp(incident)
    return vin[0]*np.asarray(modal[is_idx], dtype=complex) + vin[1]*np.asarray(modal[ip_idx], dtype=complex)


def _precompute_poynting_totals_from_modal(result, modal_key, pols=('RCP','LCP'), region_suffix='', side='reflection'):
    """Precompute total Poynting-flux maps for synthetic incident pols from b0/aN."""
    modal = result.get(modal_key)
    if modal is None:
        return {}
    pol_labels = result.get('pol_labels', [])
    params = result.get('params', {})
    energies = np.asarray(result.get('energies'), dtype=float)
    kxs = np.asarray(result.get('kxs'), dtype=float)
    kys = np.asarray(result.get('kys'), dtype=float)
    G_orders = result.get('G_orders')
    if G_orders is None:
        return {}
    nG = np.asarray(modal).shape[-1] // 2
    eps = float(params.get('n_air', 1.0))**2 if side == 'reflection' else float(params.get('n_glass', 1.5))**2
    m0 = _zero_order_index_from_G(G_orders)
    out = {}
    for pol in pols:
        maps = {
            'all': np.full((len(energies), len(kxs), len(kys)), np.nan, dtype=float),
            'zeroth': np.full((len(energies), len(kxs), len(kys)), np.nan, dtype=float),
        }
        synth = _synthesize_modal_for_incident(modal, pol_labels, pol)
        if synth is None:
            continue
        for i, Eev in enumerate(energies):
            omega = 2*np.pi*float(Eev)/HC_EV_UM
            for j, kx0 in enumerate(kxs):
                for k, ky0 in enumerate(kys):
                    amp = np.asarray(synth[i, j, k, :], dtype=complex)
                    if amp.size < 2*nG or not np.all(np.isfinite(amp)):
                        continue
                    kx, ky, kz, kz_abs, prop, nk0 = _order_geometry_np(params, G_orders, Eev, kx0, ky0, side=side)
                    q0 = kz_abs.astype(complex)
                    q = np.concatenate((q0, q0))
                    q[np.abs(q) < 1e-15] = 1e-15 + 0j
                    phi = np.eye(2*nG, dtype=complex)
                    kp = _make_kp_matrix_uniform_np(omega, eps, kx, ky)
                    if side == 'reflection':
                        a0 = _incident_a0_vector_np(pol, Eev, kx0, ky0, nG, G_orders, params.get('n_air', 1.0))
                        _, backward = _get_z_poynting_flux_np(a0, amp, omega, kp, phi, q, byorder=True)
                        orders = np.real(-backward)
                    else:
                        bN = np.zeros_like(amp)
                        forward, _ = _get_z_poynting_flux_np(amp, bN, omega, kp, phi, q, byorder=True)
                        orders = np.real(forward)
                    k0 = 2*np.pi*float(Eev)/HC_EV_UM
                    nk0_top = float(params.get('n_air', 1.0)) * k0
                    kt_inc = np.hypot(float(kx0), float(ky0))
                    cos_th = np.sqrt(max(1.0 - (kt_inc/max(nk0_top, 1e-30))**2, 0.0))
                    normalization = float(params.get('n_air', 1.0)) / max(cos_th, 1e-30)
                    orders = orders * normalization
                    maps['zeroth'][i, j, k] = float(orders[m0]) if m0 < orders.size else np.nan
                    maps['all'][i, j, k] = float(np.nansum(np.where(prop, orders, 0.0)))
        base = 'R' if side == 'reflection' else 'T'
        suf = str(region_suffix)
        out[f'{base}_{pol}_all{suf}'] = maps['all']
        out[f'{base}_{pol}_zeroth{suf}'] = maps['zeroth']
    return out


def precompute_common_maps(result):
    """Build common synthetic CP maps for faster viewer use and save them in NPZ."""
    derived = {}
    try:
        derived.update(_precompute_poynting_totals_from_modal(result, 'b0', ('RCP','LCP'), '', 'reflection'))
        derived.update(_precompute_poynting_totals_from_modal(result, 'b0_ref', ('RCP','LCP'), '_ref', 'reflection'))
        derived.update(_precompute_poynting_totals_from_modal(result, 'aN', ('RCP','LCP'), '', 'transmission'))
        derived.update(_precompute_poynting_totals_from_modal(result, 'aN_ref', ('RCP','LCP'), '_ref', 'transmission'))
        for pol in ('RCP','LCP'):
            for order in ('all','zeroth'):
                R = derived.get(f'R_{pol}_{order}')
                T = derived.get(f'T_{pol}_{order}')
                Rr = derived.get(f'R_{pol}_{order}_ref')
                Tr = derived.get(f'T_{pol}_{order}_ref')
                if R is not None and T is not None:
                    derived[f'A_{pol}_{order}'] = 1.0 - R - T
                if Rr is not None and Tr is not None:
                    derived[f'A_{pol}_{order}_ref'] = 1.0 - Rr - Tr
                if R is not None and Rr is not None:
                    derived[f'dRoverR_{pol}_{order}'] = _safe_divide_num(R - Rr, Rr)
                if T is not None and Tr is not None:
                    derived[f'dToverT_{pol}_{order}'] = _safe_divide_num(T - Tr, Tr)
                A = derived.get(f'A_{pol}_{order}')
                Ar = derived.get(f'A_{pol}_{order}_ref')
                if A is not None and Ar is not None:
                    derived[f'dAoverA_{pol}_{order}'] = _safe_divide_num(A - Ar, Ar)
        for prefix in ('R','T','A'):
            for order in ('all','zeroth'):
                rcp = derived.get(f'{prefix}_RCP_{order}')
                lcp = derived.get(f'{prefix}_LCP_{order}')
                if rcp is not None and lcp is not None:
                    derived[f'{prefix}_CD_{order}'] = _safe_divide_num(rcp - lcp, rcp + lcp)
                rcp_ref = derived.get(f'{prefix}_RCP_{order}_ref')
                lcp_ref = derived.get(f'{prefix}_LCP_{order}_ref')
                if rcp_ref is not None and lcp_ref is not None:
                    derived[f'{prefix}_CD_{order}_ref'] = _safe_divide_num(rcp_ref - lcp_ref, rcp_ref + lcp_ref)
        derived['precomputed_common_info_json'] = np.array(json.dumps({
            'description': 'Common synthetic circular maps computed in solver from saved p/s b0 and aN using RCWA Poynting flux.',
            'incident_polarizations': ['RCP','LCP'],
            'orders': ['all','zeroth'],
            'regions': ['sample','reference'],
            'convention': 'RCP=(p+i*s)/sqrt(2), same as viewer',
        }, indent=2))
    except Exception as exc:
        derived['precomputed_common_error'] = np.array(str(exc))
    return derived


def _solve_energy_block_for_parallel(args):
    """
    Top-level function for Windows multiprocessing. Computes one complete energy
    slice for all selected polarizations, kx and ky values.
    """
    _set_single_thread_blas_env()
    p, iE, E, kxs, kys, pol_labels = args
    npol, nkx, nky = len(pol_labels), len(kxs), len(kys)
    block_shape = (npol, nkx, nky)

    block = {
        "iE": iE,
        "E": E,
        "R": np.zeros(block_shape, dtype=float),
        "T": np.zeros(block_shape, dtype=float),
        "A": np.zeros(block_shape, dtype=float),
        "Rref": np.zeros(block_shape, dtype=float),
        "Tref": np.zeros(block_shape, dtype=float),
        "Aref": np.zeros(block_shape, dtype=float),
        "dRoverR": np.zeros(block_shape, dtype=float),
        "R_b_spec": np.full(block_shape, np.nan, dtype=float),
        "phase_b_spec": np.full(block_shape, np.nan, dtype=float),
        "Ex_b_spec": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "Ey_b_spec": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "Ez_b_spec": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "E_RCP_b_spec": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "E_LCP_b_spec": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "E_RCP_b_spec_ref": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "E_LCP_b_spec_ref": np.full(block_shape, np.nan + 1j*np.nan, dtype=complex),
        "b0": None, "b0_ref": None, "aN": None, "aN_ref": None,
        "E_top": None, "H_top": None, "E_bottom": None, "H_bottom": None,
        "E_top_ref": None, "H_top_ref": None, "E_bottom_ref": None, "H_bottom_ref": None,
        "G_orders": None,
    }

    for ipol, pol in enumerate(pol_labels):
        for j, kx in enumerate(kxs):
            for k, ky in enumerate(kys):
                rp, tp, ap, bp, apN, Gp, Et, Ht, Eb, Hb, rbp, phbp, exbp, eybp, ezbp, ercpbp, elcpbp = unpack_raw_result(
                    solve_patterned_rcwa_raw(p, E, kx, ky, pol)
                )
                rr, tr, ar, br, arN, Gr, Etr, Htr, Ebr, Hbr, _rbr, _phbr, _exbr, _eybr, _ezbr, ercpbr, elcpbr = unpack_raw_result(
                    solve_reference_rcwa_raw(p, E, kx, ky, pol)
                )

                block["R"][ipol, j, k] = rp
                block["T"][ipol, j, k] = tp
                block["A"][ipol, j, k] = ap
                block["Rref"][ipol, j, k] = rr
                block["Tref"][ipol, j, k] = tr
                block["Aref"][ipol, j, k] = ar
                block["dRoverR"][ipol, j, k] = (rp - rr) / rr if np.isfinite(rr) and abs(rr) > 1e-15 else np.nan
                block["R_b_spec"][ipol, j, k] = rbp
                block["phase_b_spec"][ipol, j, k] = phbp
                block["Ex_b_spec"][ipol, j, k] = exbp
                block["Ey_b_spec"][ipol, j, k] = eybp
                block["Ez_b_spec"][ipol, j, k] = ezbp
                block["E_RCP_b_spec"][ipol, j, k] = ercpbp
                block["E_LCP_b_spec"][ipol, j, k] = elcpbp
                block["E_RCP_b_spec_ref"][ipol, j, k] = ercpbr
                block["E_LCP_b_spec_ref"][ipol, j, k] = elcpbr

                if bp is not None:
                    if block["b0"] is None:
                        n_amp = len(bp)
                        block["b0"] = np.full(block_shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                        block["b0_ref"] = np.full(block_shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                    block["b0"][ipol, j, k, :len(bp)] = bp
                if br is not None:
                    if block["b0_ref"] is None:
                        n_amp = len(br)
                        block["b0_ref"] = np.full(block_shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                    block["b0_ref"][ipol, j, k, :len(br)] = br

                if apN is not None:
                    if block["aN"] is None:
                        n_amp_t = len(apN)
                        block["aN"] = np.full(block_shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                        block["aN_ref"] = np.full(block_shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                    block["aN"][ipol, j, k, :len(apN)] = apN
                if arN is not None:
                    if block["aN_ref"] is None:
                        n_amp_t = len(arN)
                        block["aN_ref"] = np.full(block_shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                    block["aN_ref"][ipol, j, k, :len(arN)] = arN

                if Et is not None:
                    if block["E_top"] is None:
                        fshape = block_shape + Et.shape
                        for name in ("E_top", "H_top", "E_bottom", "H_bottom", "E_top_ref", "H_top_ref", "E_bottom_ref", "H_bottom_ref"):
                            block[name] = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                    block["E_top"][ipol, j, k, :, :] = Et
                    block["H_top"][ipol, j, k, :, :] = Ht
                    if Eb is not None:
                        block["E_bottom"][ipol, j, k, :, :] = Eb
                        block["H_bottom"][ipol, j, k, :, :] = Hb
                    if Etr is not None:
                        block["E_top_ref"][ipol, j, k, :, :] = Etr
                        block["H_top_ref"][ipol, j, k, :, :] = Htr
                    if Ebr is not None:
                        block["E_bottom_ref"][ipol, j, k, :, :] = Ebr
                        block["H_bottom_ref"][ipol, j, k, :, :] = Hbr

                if block["G_orders"] is None and Gp is not None:
                    block["G_orders"] = Gp

    return block


class Worker(QThread):
    progress = pyqtSignal(int, int, int, int, int, str, float, float, float, float, float)
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal(object)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            p = self.params
            energies = np.linspace(p["E_min"], p["E_max"], p["E_points"])
            kxs = np.linspace(p["kx_min"], p["kx_max"], p["kx_points"])
            kys = np.linspace(p["ky_min"], p["ky_max"], p["ky_points"])
            pol_labels = pol_list_from_mode(p.get("pol", "p"))

            shape = (len(pol_labels), len(energies), len(kxs), len(kys))
            R = np.zeros(shape)
            T = np.zeros(shape)
            A = np.zeros(shape)
            Rref = np.zeros(shape)
            Tref = np.zeros(shape)
            Aref = np.zeros(shape)
            dRoverR = np.zeros(shape)
            R_b_spec = np.full(shape, np.nan, dtype=float)
            phase_b_spec = np.full(shape, np.nan, dtype=float)
            Ex_b_spec = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
            Ey_b_spec = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
            Ez_b_spec = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
            E_RCP_b_spec = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
            E_LCP_b_spec = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
            E_RCP_b_spec_ref = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
            E_LCP_b_spec_ref = np.full(shape, np.nan + 1j*np.nan, dtype=complex)

            b0 = None
            b0_ref = None
            aN = None
            aN_ref = None
            E_top = None
            H_top = None
            E_bottom = None
            H_bottom = None
            E_top_ref = None
            H_top_ref = None
            E_bottom_ref = None
            H_bottom_ref = None
            G_orders = None

            count = 0
            total = len(pol_labels) * len(energies) * len(kxs) * len(kys)
            start_time = time.perf_counter()

            def partial_payload():
                return {
                    "params": p, "pol_labels": np.array(pol_labels, dtype=object),
                    "energies": energies, "kxs": kxs, "kys": kys,
                    "R": R, "T": T, "A": A,
                    "Rref": Rref, "Tref": Tref, "Aref": Aref,
                    "dRoverR": dRoverR,
                    "R_b_spec": R_b_spec, "phase_b_spec": phase_b_spec,
                    "Ex_b_spec": Ex_b_spec, "Ey_b_spec": Ey_b_spec, "Ez_b_spec": Ez_b_spec,
                    "E_RCP_b_spec": E_RCP_b_spec, "E_LCP_b_spec": E_LCP_b_spec,
                    "E_RCP_b_spec_ref": E_RCP_b_spec_ref, "E_LCP_b_spec_ref": E_LCP_b_spec_ref,
                    "b0": b0, "b0_ref": b0_ref,
                    "aN": aN, "aN_ref": aN_ref,
                    "E_top": E_top, "H_top": H_top,
                    "E_bottom": E_bottom, "H_bottom": H_bottom,
                    "E_top_ref": E_top_ref, "H_top_ref": H_top_ref,
                    "E_bottom_ref": E_bottom_ref, "H_bottom_ref": H_bottom_ref,
                    "G_orders": G_orders,
                    "completed_points": count, "total_points": total,
                }

            if str(p.get("parallel_mode", "off")).lower().startswith("energies"):
                workers_req = int(p.get("parallel_workers", 0) or 0)
                max_workers = workers_req if workers_req > 0 else max(1, min(os.cpu_count() or 1, len(energies)))
                max_workers = max(1, min(max_workers, len(energies)))
                ctx = multiprocessing.get_context("spawn")
                futures = []
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                    for i, E in enumerate(energies):
                        futures.append(ex.submit(_solve_energy_block_for_parallel, (p, i, float(E), kxs, kys, pol_labels)))

                    done_blocks = 0
                    for fut in as_completed(futures):
                        if self._stop_requested:
                            for f in futures:
                                f.cancel()
                            self.cancelled.emit(partial_payload())
                            return

                        block = fut.result()
                        i = int(block["iE"])

                        R[:, i, :, :] = block["R"]
                        T[:, i, :, :] = block["T"]
                        A[:, i, :, :] = block["A"]
                        Rref[:, i, :, :] = block["Rref"]
                        Tref[:, i, :, :] = block["Tref"]
                        Aref[:, i, :, :] = block["Aref"]
                        dRoverR[:, i, :, :] = block["dRoverR"]
                        R_b_spec[:, i, :, :] = block["R_b_spec"]
                        phase_b_spec[:, i, :, :] = block["phase_b_spec"]
                        Ex_b_spec[:, i, :, :] = block["Ex_b_spec"]
                        Ey_b_spec[:, i, :, :] = block["Ey_b_spec"]
                        Ez_b_spec[:, i, :, :] = block["Ez_b_spec"]
                        E_RCP_b_spec[:, i, :, :] = block["E_RCP_b_spec"]
                        E_LCP_b_spec[:, i, :, :] = block["E_LCP_b_spec"]
                        E_RCP_b_spec_ref[:, i, :, :] = block["E_RCP_b_spec_ref"]
                        E_LCP_b_spec_ref[:, i, :, :] = block["E_LCP_b_spec_ref"]

                        if block["b0"] is not None:
                            if b0 is None:
                                n_amp = block["b0"].shape[-1]
                                b0 = np.full(shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                                b0_ref = np.full(shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                            b0[:, i, :, :, :] = block["b0"]
                            if block["b0_ref"] is not None:
                                b0_ref[:, i, :, :, :] = block["b0_ref"]

                        if block["aN"] is not None:
                            if aN is None:
                                n_amp_t = block["aN"].shape[-1]
                                aN = np.full(shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                                aN_ref = np.full(shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                            aN[:, i, :, :, :] = block["aN"]
                            if block["aN_ref"] is not None:
                                aN_ref[:, i, :, :, :] = block["aN_ref"]

                        if block["E_top"] is not None:
                            if E_top is None:
                                fshape = shape + block["E_top"].shape[-2:]
                                E_top = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                H_top = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                E_bottom = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                H_bottom = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                E_top_ref = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                H_top_ref = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                E_bottom_ref = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                                H_bottom_ref = np.full(fshape, np.nan + 1j*np.nan, dtype=complex)
                            E_top[:, i, :, :, :, :] = block["E_top"]
                            H_top[:, i, :, :, :, :] = block["H_top"]
                            E_bottom[:, i, :, :, :, :] = block["E_bottom"]
                            H_bottom[:, i, :, :, :, :] = block["H_bottom"]
                            E_top_ref[:, i, :, :, :, :] = block["E_top_ref"]
                            H_top_ref[:, i, :, :, :, :] = block["H_top_ref"]
                            E_bottom_ref[:, i, :, :, :, :] = block["E_bottom_ref"]
                            H_bottom_ref[:, i, :, :, :, :] = block["H_bottom_ref"]

                        if G_orders is None and block["G_orders"] is not None:
                            G_orders = block["G_orders"]

                        done_blocks += 1
                        count = done_blocks * len(pol_labels) * len(kxs) * len(kys)
                        pct = int(round(100 * count / max(total, 1)))
                        elapsed = time.perf_counter() - start_time
                        eta = (elapsed / max(done_blocks, 1)) * (len(energies) - done_blocks)
                        self.progress.emit(count, total, pct, done_blocks, len(energies), f"parallel {max_workers} workers", block["E"], float("nan"), float("nan"), elapsed, eta)

                self.done.emit({
                    "params": p, "pol_labels": np.array(pol_labels, dtype=object),
                    "energies": energies, "kxs": kxs, "kys": kys,
                    "R": R, "T": T, "A": A,
                    "Rref": Rref, "Tref": Tref, "Aref": Aref,
                    "dRoverR": dRoverR,
                    "R_b_spec": R_b_spec, "phase_b_spec": phase_b_spec,
                    "Ex_b_spec": Ex_b_spec, "Ey_b_spec": Ey_b_spec, "Ez_b_spec": Ez_b_spec,
                    "E_RCP_b_spec": E_RCP_b_spec, "E_LCP_b_spec": E_LCP_b_spec,
                    "E_RCP_b_spec_ref": E_RCP_b_spec_ref, "E_LCP_b_spec_ref": E_LCP_b_spec_ref,
                    "b0": b0, "b0_ref": b0_ref,
                    "aN": aN, "aN_ref": aN_ref,
                    "E_top": E_top, "H_top": H_top,
                    "E_bottom": E_bottom, "H_bottom": H_bottom,
                    "E_top_ref": E_top_ref, "H_top_ref": H_top_ref,
                    "E_bottom_ref": E_bottom_ref, "H_bottom_ref": H_bottom_ref,
                    "G_orders": G_orders,
                })
                return

            for ipol, pol in enumerate(pol_labels):
                for i, E in enumerate(energies):
                    for j, kx in enumerate(kxs):
                        for k, ky in enumerate(kys):
                            if self._stop_requested:
                                self.cancelled.emit(partial_payload())
                                return

                            rp, tp, ap, bp, apN, Gp, Et, Ht, Eb, Hb, rbp, phbp, exbp, eybp, ezbp, ercpbp, elcpbp = unpack_raw_result(
                                solve_patterned_rcwa_raw(p, E, kx, ky, pol)
                            )
                            rr, tr, ar, br, arN, Gr, Etr, Htr, Ebr, Hbr, _rbr, _phbr, _exbr, _eybr, _ezbr, ercpbr, elcpbr = unpack_raw_result(
                                solve_reference_rcwa_raw(p, E, kx, ky, pol)
                            )

                            R[ipol, i, j, k] = rp
                            T[ipol, i, j, k] = tp
                            A[ipol, i, j, k] = ap
                            Rref[ipol, i, j, k] = rr
                            Tref[ipol, i, j, k] = tr
                            Aref[ipol, i, j, k] = ar
                            dRoverR[ipol, i, j, k] = (rp - rr) / rr if np.isfinite(rr) and abs(rr) > 1e-15 else np.nan
                            R_b_spec[ipol, i, j, k] = rbp
                            phase_b_spec[ipol, i, j, k] = phbp
                            Ex_b_spec[ipol, i, j, k] = exbp
                            Ey_b_spec[ipol, i, j, k] = eybp
                            Ez_b_spec[ipol, i, j, k] = ezbp
                            E_RCP_b_spec[ipol, i, j, k] = ercpbp
                            E_LCP_b_spec[ipol, i, j, k] = elcpbp
                            E_RCP_b_spec_ref[ipol, i, j, k] = ercpbr
                            E_LCP_b_spec_ref[ipol, i, j, k] = elcpbr

                            if bp is not None:
                                if b0 is None:
                                    n_amp = len(bp)
                                    b0 = np.full(shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                                    b0_ref = np.full(shape + (n_amp,), np.nan + 1j*np.nan, dtype=complex)
                                b0[ipol, i, j, k, :len(bp)] = bp

                            if br is not None and b0_ref is not None:
                                b0_ref[ipol, i, j, k, :len(br)] = br

                            if apN is not None:
                                if aN is None:
                                    n_amp_t = len(apN)
                                    aN = np.full(shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                                    aN_ref = np.full(shape + (n_amp_t,), np.nan + 1j*np.nan, dtype=complex)
                                aN[ipol, i, j, k, :len(apN)] = apN

                            if arN is not None and aN_ref is not None:
                                aN_ref[ipol, i, j, k, :len(arN)] = arN

                            if Et is not None:
                                if E_top is None:
                                    field_shape = shape + Et.shape
                                    E_top = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    H_top = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    E_bottom = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    H_bottom = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    E_top_ref = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    H_top_ref = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    E_bottom_ref = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)
                                    H_bottom_ref = np.full(field_shape, np.nan + 1j*np.nan, dtype=complex)

                                E_top[ipol, i, j, k, :, :] = Et
                                H_top[ipol, i, j, k, :, :] = Ht
                                if Eb is not None:
                                    E_bottom[ipol, i, j, k, :, :] = Eb
                                    H_bottom[ipol, i, j, k, :, :] = Hb
                                if Etr is not None:
                                    E_top_ref[ipol, i, j, k, :, :] = Etr
                                    H_top_ref[ipol, i, j, k, :, :] = Htr
                                if Ebr is not None:
                                    E_bottom_ref[ipol, i, j, k, :, :] = Ebr
                                    H_bottom_ref[ipol, i, j, k, :, :] = Hbr

                            if G_orders is None and Gp is not None:
                                G_orders = Gp

                            count += 1
                            pct = int(round(100 * count / max(total, 1)))
                            elapsed = time.perf_counter() - start_time
                            eta = (elapsed / count) * (total - count) if count > 0 else float("nan")
                            self.progress.emit(count, total, pct, ipol + 1, len(pol_labels), pol, E, kx, ky, elapsed, eta)

            self.done.emit({
                "params": p, "pol_labels": np.array(pol_labels, dtype=object),
                "energies": energies, "kxs": kxs, "kys": kys,
                "R": R, "T": T, "A": A,
                "Rref": Rref, "Tref": Tref, "Aref": Aref,
                "dRoverR": dRoverR,
                "R_b_spec": R_b_spec, "phase_b_spec": phase_b_spec,
                "Ex_b_spec": Ex_b_spec, "Ey_b_spec": Ey_b_spec, "Ez_b_spec": Ez_b_spec,
                "E_RCP_b_spec": E_RCP_b_spec, "E_LCP_b_spec": E_LCP_b_spec,
                "E_RCP_b_spec_ref": E_RCP_b_spec_ref, "E_LCP_b_spec_ref": E_LCP_b_spec_ref,
                "b0": b0, "b0_ref": b0_ref,
                "aN": aN, "aN_ref": aN_ref,
                "E_top": E_top, "H_top": H_top,
                "E_bottom": E_bottom, "H_bottom": H_bottom,
                "E_top_ref": E_top_ref, "H_top_ref": H_top_ref,
                "E_bottom_ref": E_bottom_ref, "H_bottom_ref": H_bottom_ref,
                "G_orders": G_orders,
            })
        except Exception:
            self.error.emit(traceback.format_exc())

class RCWAQtGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"lighTTice Solver v{SOLVER_VERSION} — periodic structures")
        self.resize(1520, 930)
        self.result = None
        self.worker = None
        self.inputs = {}
        self.top_layers: list[TopLayer] = []
        self.gold_file_material = Material(mode="file")
        self.cell_file_path = ""
        self.cell_design = None

        self.build_ui()
        self.update_layer_table()
        self.update_geometry_preview()

    def build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_container)
        splitter.addWidget(scroll)

        form = QFormLayout()
        controls_layout.addLayout(form)

        defaults = {
            "period_x": "0.680", "period_y": "0.680",
            "slit_short": "0.050", "slit_long": "0.300",
            "gold_thickness": "0.200",
            "n_air": "1.0", "n_glass": "1.5",
            "rotation_step": "30",
            "n_cells_x": "6", "n_cells_y": "1",
            "Nx": "160", "Ny": "160", "nG": "30",
            "E_min": "1.5", "E_max": "2.0", "E_points": "100",
            "kx_min": "-5.25", "kx_max": "5.25", "kx_points": "100",
            "ky_min": "0", "ky_max": "0", "ky_points": "1",
        }

        def make_group(title, items, width=92):
            box = QGroupBox(title)
            layout = QFormLayout(box)
            layout.setLabelAlignment(Qt.AlignRight)
            layout.setFormAlignment(Qt.AlignTop)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setVerticalSpacing(4)
            for key, label in items:
                edit = QLineEdit(defaults[key])
                edit.setMaximumWidth(width)
                self.inputs[key] = edit
                layout.addRow(label, edit)
            return box

        # v1.3 compact control layout:
        # left column  = Geometry + Solver output
        # right column = Materials/media + Scan/accuracy
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        controls_layout.addWidget(top_container)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(6)
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(6)

        top_layout.addLayout(left_col, 1)
        top_layout.addLayout(right_col, 1)

        geometry_box = make_group("Geometry / macrocell", [
            ("period_x", "Period x, um"),
            ("period_y", "Period y, um"),
            ("slit_short", "Slit short, um"),
            ("slit_long", "Slit long, um"),
            ("gold_thickness", "Au thickness, um"),
            ("rotation_step", "Rot. step, deg"),
            ("n_cells_x", "Macro cells x"),
            ("n_cells_y", "Macro cells y"),
        ])
        left_col.addWidget(geometry_box)

        cell_box = QGroupBox("Designer cell file")
        cell_layout = QGridLayout(cell_box)
        self.cell_file_edit = QLineEdit("")
        self.cell_file_edit.setReadOnly(True)
        self.cell_file_edit.setPlaceholderText("No external cell file: using built-in slit macrocell")
        load_cell_btn = QPushButton("Load cell JSON")
        clear_cell_btn = QPushButton("Clear")
        load_cell_btn.clicked.connect(self.load_cell_file)
        clear_cell_btn.clicked.connect(self.clear_cell_file)
        cell_layout.addWidget(QLabel("Cell"), 0, 0)
        cell_layout.addWidget(self.cell_file_edit, 0, 1)
        cell_layout.addWidget(load_cell_btn, 1, 0)
        cell_layout.addWidget(clear_cell_btn, 1, 1)
        cell_box.setToolTip("Load a cell JSON exported from the RCWA Designer. When loaded, the patterned Au layer is rasterized from the file instead of the built-in slit array.")
        left_col.addWidget(cell_box)

        solver_box = QGroupBox("Solver output")
        solver_form = QFormLayout(solver_box)
        solver_form.setLabelAlignment(Qt.AlignRight)
        solver_form.setContentsMargins(8, 8, 8, 8)
        solver_form.setVerticalSpacing(4)

        self.pol_box = QComboBox()
        self.pol_box.addItems(["p and s", "p", "s"])
        self.pol_box.setMaximumWidth(130)
        self.pol_box.setToolTip(
            "Linear-basis export. Recommended/default: p and s. Viewer synthesizes RCP/LCP from saved complex b0_p/b0_s and transmitted aN_p/aN_s."
        )
        solver_form.addRow("Pol. basis", self.pol_box)

        self.reference_mode_box = QComboBox()
        self.reference_mode_box.addItems([
            "uniform_gold",
            "patterned_gold",
            "no_gold",
        ])
        self.reference_mode_box.setMaximumWidth(130)
        self.reference_mode_box.setToolTip(
            "Reference used for quick-check ΔR/R and saved in NPZ.\n"
            "uniform_gold: selected reference layers + continuous Au film.\n"
            "patterned_gold: selected reference layers + same slit array.\n"
            "no_gold: selected reference layers on glass, no Au."
        )
        solver_form.addRow("Reference", self.reference_mode_box)

        self.field_preview_enable = QCheckBox("single point only")
        self.field_preview_enable.setChecked(True)
        self.field_preview_enable.setToolTip("Preview the complex real-space field for one energy and one k-point. This does not run/save the full scan field volume.")
        solver_form.addRow("Field preview", self.field_preview_enable)

        self.field_E_edit = QLineEdit(defaults["E_min"]); self.field_E_edit.setMaximumWidth(92)
        self.field_kx_edit = QLineEdit("0.0"); self.field_kx_edit.setMaximumWidth(92)
        self.field_ky_edit = QLineEdit("0.0"); self.field_ky_edit.setMaximumWidth(92)
        self.field_zfrac_edit = QLineEdit("0.50"); self.field_zfrac_edit.setMaximumWidth(92)
        self.field_layer_box = QComboBox()
        self.field_layer_box.addItems(["gold slit layer", "top air", "bottom glass"])
        self.field_layer_box.setMaximumWidth(130)
        solver_form.addRow("Field E, eV", self.field_E_edit)
        solver_form.addRow("Field kx", self.field_kx_edit)
        solver_form.addRow("Field ky", self.field_ky_edit)
        solver_form.addRow("Field z frac", self.field_zfrac_edit)
        solver_form.addRow("Field layer", self.field_layer_box)

        # Derivative plotting belongs in the viewer. The solver only shows quick-check R and ΔR/R.
        self.derivative_source_box = QComboBox()
        self.derivative_source_box.addItems(["dR/dE of R"])
        self.derivative_source_box.setVisible(False)

        left_col.addWidget(solver_box)
        left_col.addStretch(0)

        media_box = make_group("Materials / media", [
            ("n_air", "Incident n"),
            ("n_glass", "Glass n"),
        ])
        right_col.addWidget(media_box)

        gold_box = QGroupBox("Gold / Au optical constants")
        gold_form = QFormLayout(gold_box)
        gold_form.setLabelAlignment(Qt.AlignRight)
        gold_form.setContentsMargins(8, 8, 8, 8)
        gold_form.setVerticalSpacing(4)

        self.gold_mode_box = QComboBox()
        self.gold_mode_box.addItems(["built-in Drude", "spectral file"])
        self.gold_mode_box.setMaximumWidth(150)
        self.gold_mode_box.setToolTip(
            "built-in Drude: original solver model.\n"
            "spectral file: load Au n,k or epsilon versus energy/wavelength, using the same mapper as top layers."
        )
        gold_form.addRow("Au model", self.gold_mode_box)

        self.gold_file_edit = QLineEdit("")
        self.gold_file_edit.setReadOnly(True)
        gold_browse_btn = QPushButton("Browse/map")
        gold_browse_btn.clicked.connect(self.browse_gold_file)
        gold_remap_btn = QPushButton("Remap/plot")
        gold_remap_btn.clicked.connect(self.remap_gold_file)
        gold_file_row = QHBoxLayout()
        gold_file_row.addWidget(self.gold_file_edit, 1)
        gold_file_row.addWidget(gold_browse_btn)
        gold_file_row.addWidget(gold_remap_btn)
        gold_form.addRow("Au file", gold_file_row)

        self.gold_info_label = QLabel("Default: spectral Drude-like Au from built-in formula; choose spectral file for measured Au data.")
        self.gold_info_label.setWordWrap(True)
        gold_form.addRow("", self.gold_info_label)
        right_col.addWidget(gold_box)

        scan_box = make_group("Scan / accuracy", [
            ("E_min", "E min, eV"),
            ("E_max", "E max, eV"),
            ("E_points", "E points"),
            ("kx_min", "kx min"),
            ("kx_max", "kx max"),
            ("kx_points", "kx points"),
            ("ky_min", "ky min"),
            ("ky_max", "ky max"),
            ("ky_points", "ky points"),
            ("Nx", "Pixels Nx"),
            ("Ny", "Pixels Ny"),
            ("nG", "Fourier nG"),
        ])
        right_col.addWidget(scan_box)

        parallel_box = QGroupBox("Parallel / ETA")
        parallel_form = QFormLayout(parallel_box)
        parallel_form.setLabelAlignment(Qt.AlignRight)
        parallel_form.setContentsMargins(8, 8, 8, 8)
        parallel_form.setVerticalSpacing(4)
        self.parallel_mode_box = QComboBox()
        self.parallel_mode_box.addItems(["off", "energies"])
        self.parallel_mode_box.setToolTip("off = sequential scan. energies = split energy slices across CPU worker processes.")
        self.parallel_workers_edit = QLineEdit("auto")
        self.parallel_workers_edit.setMaximumWidth(92)
        self.parallel_workers_edit.setToolTip("Number of CPU workers. Use auto, or an integer such as 2, 4, 8.")
        parallel_form.addRow("Parallel mode", self.parallel_mode_box)
        parallel_form.addRow("CPU workers", self.parallel_workers_edit)
        left_col.addWidget(parallel_box)
        left_col.addStretch(0)

        note = QLabel("Nx/Ny = geometry pixels over the whole macrocell; nG = RCWA Fourier basis. Save NPZ writes a self-describing file.")
        note.setWordWrap(True)
        controls_layout.addWidget(note)

        layer_box = QGroupBox("Homogeneous top layers: top to bottom")
        layer_layout = QVBoxLayout(layer_box)
        controls_layout.addWidget(layer_box)

        self.layer_table = QTableWidget(0, 6)
        self.layer_table.setMaximumHeight(130)
        self.layer_table.setHorizontalHeaderLabels(["#", "Name", "d (um)", "Ref", "Material", "File"])
        self.layer_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.layer_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layer_layout.addWidget(self.layer_table)

        lbtns1 = QHBoxLayout()
        lbtns2 = QHBoxLayout()
        add_btn = QPushButton("Add")
        edit_btn = QPushButton("Edit")
        dup_btn = QPushButton("Duplicate")
        rem_btn = QPushButton("Remove")
        up_btn = QPushButton("Up")
        down_btn = QPushButton("Down")
        load_btn = QPushButton("Load layers")
        save_btn = QPushButton("Save layers")
        add_btn.clicked.connect(self.add_layer)
        edit_btn.clicked.connect(self.edit_layer)
        dup_btn.clicked.connect(self.duplicate_layer)
        rem_btn.clicked.connect(self.remove_layer)
        up_btn.clicked.connect(self.move_layer_up)
        down_btn.clicked.connect(self.move_layer_down)
        load_btn.clicked.connect(self.load_layers)
        save_btn.clicked.connect(self.save_layers)
        for b in [add_btn, edit_btn, dup_btn, rem_btn]:
            lbtns1.addWidget(b)
        for b in [up_btn, down_btn, load_btn, save_btn]:
            lbtns2.addWidget(b)
        layer_layout.addLayout(lbtns1)
        layer_layout.addLayout(lbtns2)

        preview_row = QHBoxLayout()
        preview_btn = QPushButton("Preview geometry")
        preview_btn.clicked.connect(self.update_geometry_preview)
        field_preview_btn = QPushButton("Preview field")
        field_preview_btn.clicked.connect(self.preview_single_field)
        preview_row.addWidget(preview_btn)
        preview_row.addWidget(field_preview_btn)
        controls_layout.addLayout(preview_row)

        run_row = QHBoxLayout()
        run_btn = QPushButton("Run RCWA")
        run_btn.clicked.connect(self.run_simulation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        run_row.addWidget(run_btn)
        run_row.addWidget(self.stop_btn)
        controls_layout.addLayout(run_row)

        save_row = QHBoxLayout()
        save_npz_btn = QPushButton("Save NPZ")
        save_npz_btn.clicked.connect(self.save_npz)
        save_csv_btn = QPushButton("Save CSV")
        save_csv_btn.clicked.connect(self.save_csv)
        save_row.addWidget(save_npz_btn)
        save_row.addWidget(save_csv_btn)
        controls_layout.addLayout(save_row)

        config_row = QHBoxLayout()
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_config)
        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_config)
        config_row.addWidget(save_config_btn)
        config_row.addWidget(load_config_btn)
        controls_layout.addLayout(config_row)

        self.progress = QProgressBar()
        controls_layout.addWidget(self.progress)
        self.status = QLabel("Ready")
        controls_layout.addWidget(self.status)
        controls_layout.addStretch(1)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        splitter.addWidget(plot_widget)

        self.fig = Figure(figsize=(16, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # Initial preview layout: keep the solver window consistent before and after solving.
        # The heavy polarization analysis is now done in the viewer, so the solver only
        # shows geometry + simple p/s reflectance maps.
        self.fig.clear()
        self.ax_geom = self.fig.add_subplot(1, 3, 1)
        self.ax_p_preview = self.fig.add_subplot(1, 3, 2)
        self.ax_s_preview = self.fig.add_subplot(1, 3, 3)
        self.ax_geom.set_title("Geometry preview")
        self.ax_p_preview.set_title("R, p excitation")
        self.ax_s_preview.set_title("R, s excitation")
        self.ax_geom.text(0.5, 0.5, "Click Preview Geometry", ha="center", va="center", transform=self.ax_geom.transAxes)
        self.ax_p_preview.text(0.5, 0.5, "Run simulation", ha="center", va="center", transform=self.ax_p_preview.transAxes)
        self.ax_s_preview.text(0.5, 0.5, "Run simulation", ha="center", va="center", transform=self.ax_s_preview.transAxes)
        for ax in (self.ax_geom, self.ax_p_preview, self.ax_s_preview):
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()

        splitter.setSizes([520, 1000])

    def browse_gold_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Au material file", "", "Data files (*.csv *.txt *.dat);;All files (*)"
        )
        if not path:
            return
        self.gold_file_material = Material(mode="file", file_path=path)
        dlg = MaterialFileMapperDialog(self, path, self.gold_file_material)
        if dlg.exec_() == QDialog.Accepted and dlg.result is not None:
            self.gold_file_material = dlg.result
            self.gold_file_material.mode = "file"
            self.gold_file_edit.setText(self.gold_file_material.file_path)
            self.gold_mode_box.setCurrentText("spectral file")
            self.gold_info_label.setText("Au spectral file loaded: " + self.gold_file_material.file_path)

    def remap_gold_file(self):
        path = self.gold_file_edit.text().strip() or self.gold_file_material.file_path
        if not path:
            QMessageBox.information(self, "No Au file", "Select an Au material file first.")
            return
        self.gold_file_material.file_path = path
        self.gold_file_material.mode = "file"
        dlg = MaterialFileMapperDialog(self, path, self.gold_file_material)
        if dlg.exec_() == QDialog.Accepted and dlg.result is not None:
            self.gold_file_material = dlg.result
            self.gold_file_material.mode = "file"
            self.gold_file_edit.setText(self.gold_file_material.file_path)
            self.gold_mode_box.setCurrentText("spectral file")
            self.gold_info_label.setText("Au spectral file loaded: " + self.gold_file_material.file_path)

    def selected_layer_index(self):
        rows = sorted(set(i.row() for i in self.layer_table.selectedIndexes()))
        return rows[0] if rows else None

    def material_summary(self, m: Material):
        if m.mode == "n":
            return f"n={m.n_value.real:g}+{m.n_value.imag:g}i"
        if m.mode == "eps":
            return f"eps={m.eps_value.real:g}+{m.eps_value.imag:g}i"
        return f"file ({m.file_data_kind}, {m.file_x_kind})"

    def update_layer_table(self):
        self.layer_table.setRowCount(len(self.top_layers))
        for i, layer in enumerate(self.top_layers):
            file_text = layer.material.file_path if layer.material.mode == "file" else ""
            vals = [str(i), layer.name, f"{layer.thickness_um:g}", "yes" if layer.use_in_reference else "no", self.material_summary(layer.material), file_text]
            for j, val in enumerate(vals):
                self.layer_table.setItem(i, j, QTableWidgetItem(val))

    def add_layer(self):
        dlg = LayerDialog(self)
        if dlg.exec_() == QDialog.Accepted and dlg.result is not None:
            self.top_layers.append(dlg.result)
            self.update_layer_table()

    def edit_layer(self):
        idx = self.selected_layer_index()
        if idx is None:
            QMessageBox.information(self, "Select layer", "Select a layer first.")
            return
        dlg = LayerDialog(self, self.top_layers[idx])
        if dlg.exec_() == QDialog.Accepted and dlg.result is not None:
            self.top_layers[idx] = dlg.result
            self.update_layer_table()

    def duplicate_layer(self):
        idx = self.selected_layer_index()
        if idx is None:
            QMessageBox.information(self, "Select layer", "Select a layer first.")
            return
        self.top_layers.insert(idx + 1, TopLayer.from_dict(self.top_layers[idx].to_dict()))
        self.update_layer_table()

    def remove_layer(self):
        idx = self.selected_layer_index()
        if idx is None:
            QMessageBox.information(self, "Select layer", "Select a layer first.")
            return
        self.top_layers.pop(idx)
        self.update_layer_table()

    def move_layer_up(self):
        idx = self.selected_layer_index()
        if idx is None or idx == 0:
            return
        self.top_layers[idx - 1], self.top_layers[idx] = self.top_layers[idx], self.top_layers[idx - 1]
        self.update_layer_table()
        self.layer_table.selectRow(idx - 1)

    def move_layer_down(self):
        idx = self.selected_layer_index()
        if idx is None or idx >= len(self.top_layers) - 1:
            return
        self.top_layers[idx + 1], self.top_layers[idx] = self.top_layers[idx], self.top_layers[idx + 1]
        self.update_layer_table()
        self.layer_table.selectRow(idx + 1)

    def save_layers(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save top layers", "top_layers.json", "JSON file (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([layer.to_dict() for layer in self.top_layers], f, indent=2)
        QMessageBox.information(self, "Saved", f"Saved top layers to:\n{path}")

    def load_layers(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load top layers", "", "JSON file (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.top_layers = [TopLayer.from_dict(x) for x in data]
            self.update_layer_table()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def load_cell_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load designer cell JSON",
            "",
            "Cell JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            design = load_cell_design_json(path)
            self.cell_design = design
            self.cell_file_path = path
            self.cell_file_edit.setText(path)
            # Keep the visible dimensions consistent with the loaded cell.
            self.inputs["period_x"].setText(str(float(design.get("Lx"))))
            self.inputs["period_y"].setText(str(float(design.get("Ly"))))
            self.inputs["n_cells_x"].setText("1")
            self.inputs["n_cells_y"].setText("1")
            self.status.setText(f"Loaded designer cell: {Path(path).name}")
            self.update_geometry_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load cell error", str(e))

    def clear_cell_file(self):
        self.cell_design = None
        self.cell_file_path = ""
        if hasattr(self, "cell_file_edit"):
            self.cell_file_edit.clear()
        self.status.setText("External cell cleared: using built-in slit macrocell.")
        self.update_geometry_preview()

    def current_config_dict(self):
        """Return a JSON-serializable dictionary containing all simulation settings."""
        inputs = {key: widget.text() for key, widget in self.inputs.items()}
        return {
            "version": CONFIG_VERSION,
            "solver_version": SOLVER_VERSION,
            "app": "RCWA_solver",
            "inputs": inputs,
            "polarization": self.pol_box.currentText(),
            "reference_mode": self.reference_mode_box.currentText(),
            "derivative_mode": self.derivative_source_box.currentText(),
            "gold_material_mode": self.gold_mode_box.currentText(),
            "gold_material": self.gold_file_material.to_dict(),
            "cell_file_path": self.cell_file_path,
            "cell_design": self.cell_design,
            "top_layers": [layer.to_dict() for layer in self.top_layers],
        }

    def apply_config_dict(self, cfg):
        """Apply a configuration dictionary to the GUI."""
        inputs = cfg.get("inputs", {})
        for key, value in inputs.items():
            if key in self.inputs:
                self.inputs[key].setText(str(value))

        pol = cfg.get("polarization", "p")
        idx = self.pol_box.findText(pol)
        if idx >= 0:
            self.pol_box.setCurrentIndex(idx)

        ref_mode = cfg.get("reference_mode", "uniform_gold")
        ridx = self.reference_mode_box.findText(ref_mode)
        if ridx >= 0:
            self.reference_mode_box.setCurrentIndex(ridx)

        deriv_mode = cfg.get("derivative_mode", "dR/dE of R")
        didx = self.derivative_source_box.findText(deriv_mode)
        if didx >= 0:
            self.derivative_source_box.setCurrentIndex(didx)

        gold_mode = cfg.get("gold_material_mode", "built-in Drude")
        gidx = self.gold_mode_box.findText(gold_mode)
        if gidx >= 0:
            self.gold_mode_box.setCurrentIndex(gidx)
        self.gold_file_material = Material.from_dict(cfg.get("gold_material", {"mode": "file"}))
        if self.gold_file_material.file_path:
            self.gold_file_edit.setText(self.gold_file_material.file_path)
            self.gold_info_label.setText("Au spectral file loaded: " + self.gold_file_material.file_path)

        self.cell_file_path = cfg.get("cell_file_path", "") or ""
        self.cell_design = cfg.get("cell_design", None)
        if hasattr(self, "cell_file_edit"):
            if self.cell_design:
                self.cell_file_edit.setText(self.cell_file_path or "<embedded cell design from config>")
            else:
                self.cell_file_edit.clear()

        self.top_layers = [TopLayer.from_dict(x) for x in cfg.get("top_layers", [])]
        self.update_layer_table()
        self.update_geometry_preview()
        self.status.setText("Configuration loaded")

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save full simulation configuration",
            "rcwa_configuration.json",
            "JSON file (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.current_config_dict(), f, indent=2)
            QMessageBox.information(self, "Saved", f"Saved full configuration to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save configuration error", str(e))

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load full simulation configuration",
            "",
            "JSON file (*.json);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.apply_config_dict(cfg)
            QMessageBox.information(self, "Loaded", f"Loaded configuration from:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Load configuration error", str(e))

    def get_params(self):
        p = {}
        for key, widget in self.inputs.items():
            p[key] = float(widget.text())

        for key in ["n_cells_x", "n_cells_y", "Nx", "Ny", "nG", "E_points", "kx_points", "ky_points"]:
            p[key] = int(round(p[key]))

        p["pol"] = self.pol_box.currentText()
        p["reference_mode"] = self.reference_mode_box.currentText()
        p["derivative_mode"] = self.derivative_source_box.currentText()
        p["parallel_mode"] = self.parallel_mode_box.currentText() if hasattr(self, "parallel_mode_box") else "off"
        wtxt = self.parallel_workers_edit.text().strip().lower() if hasattr(self, "parallel_workers_edit") else "auto"
        if wtxt in ("", "auto", "0"):
            p["parallel_workers"] = 0
        else:
            p["parallel_workers"] = max(1, int(wtxt))
        p["gold_material_mode"] = self.gold_mode_box.currentText()
        if p["gold_material_mode"] == "spectral file":
            mat = Material.from_dict(self.gold_file_material.to_dict())
            mat.mode = "file"
            mat.file_path = self.gold_file_edit.text().strip() or mat.file_path
            mat.load_arrays()  # validate before starting the RCWA scan
            p["gold_material"] = mat
        else:
            p["gold_material"] = None
        p["top_layers"] = [TopLayer.from_dict(layer.to_dict()) for layer in self.top_layers]
        p["cell_file_path"] = self.cell_file_path
        p["cell_design"] = self.cell_design
        return p

    def auto_repeat_cells(self):
        try:
            step = float(self.inputs["rotation_step"].text())
            n = repeat_count_from_rotation_step(step)
            self.inputs["n_cells_x"].setText(str(n))
            self.status.setText(
                f"Macro cells x set to {n}: rotation sequence closes after {n} cells for {step:g}° step."
            )
            self.update_geometry_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_geometry_preview(self):
        try:
            p = self.get_params()
            eps, Lx, Ly = make_pattern_eps(p, -10 + 1j, p["n_air"]**2)
            mask = np.real(eps) < 0
            self.ax_geom.clear()
            plot_lattice_array(self.ax_geom, mask.astype(float), p, title="Geometry preview")
            layer_text = f"Reference: {self.reference_mode_box.currentText()}\nTop layers:\n" + ("\n".join([f"{i}: {l.name}, {l.thickness_um:g} um, ref={'yes' if l.use_in_reference else 'no'}" for i, l in enumerate(self.top_layers)]) if self.top_layers else "none")
            self.ax_geom.text(1.02, 0.98, layer_text, transform=self.ax_geom.transAxes, va="top", fontsize=8)
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            self.status.setText(f"Preview error: {e}")

    def preview_single_field(self):
        """Compute and plot a single-energy real-space complex field preview."""
        try:
            params = self.get_params()
            E0 = float(self.field_E_edit.text())
            kx0 = float(self.field_kx_edit.text())
            ky0 = float(self.field_ky_edit.text())
            zfrac = float(self.field_zfrac_edit.text())
            pol_labels = pol_list_from_mode(params.get("pol", "p"))
            pol = pol_labels[0] if pol_labels else "p"
            layer_text = self.field_layer_box.currentText()
            if layer_text.startswith("top"):
                which = "top"
            elif layer_text.startswith("bottom"):
                which = "bottom"
            else:
                which = "gold"

            self.status.setText(f"Computing field preview: pol={pol}, E={E0:g} eV, kx={kx0:g}, ky={ky0:g}...")
            QApplication.processEvents()

            data = solve_patterned_field_on_grid(
                params, E0, kx0, ky0, pol, which=which, z_fraction=zfrac,
                Nxy=[int(params["Nx"]), int(params["Ny"])]
            )

            Ex, Ey, Ez = data["Ex"], data["Ey"], data["Ez"]
            Eabs2 = data["Eabs2"]
            phase = np.angle(Ex)
            extent = [0.0, data["Lx"], 0.0, data["Ly"]]

            # Field preview uses its own layout.  Older versions expected axes named
            # ax_R/ax_dR/ax_deriv, but the current solver preview only has
            # geometry + p/s R axes.  Rebuild the figure here so Preview field works
            # both before and after a full RCWA run.
            self.fig.clear()
            ax_geom = self.fig.add_subplot(1, 4, 1)
            ax_E = self.fig.add_subplot(1, 4, 2)
            ax_phase = self.fig.add_subplot(1, 4, 3)
            ax_info = self.fig.add_subplot(1, 4, 4)

            # Keep ax_geom valid for Preview geometry after this call.
            self.ax_geom = ax_geom

            eps_plot = np.real(data["eps_grid"])
            im_geom = plot_lattice_array(ax_geom, eps_plot, params, title="Geometry / permittivity")

            im0 = plot_lattice_array(ax_E, Eabs2, params, title=r"Real-space $|E|^2$")
            self.fig.colorbar(im0, ax=ax_E, fraction=0.046, pad=0.04)

            im1 = plot_lattice_array(ax_phase, phase, params, title=r"Phase of $E_x$", vmin=-np.pi, vmax=np.pi)
            self.fig.colorbar(im1, ax=ax_phase, fraction=0.046, pad=0.04)

            ax_info.axis("off")
            ax_info.text(
                0.02, 0.96,
                f"Single-point complex field preview\n"
                f"pol = {data['pol']}\n"
                f"E = {data['E_eV']:.4g} eV\n"
                f"kx = {data['kx']:.4g} um^-1, ky = {data['ky']:.4g} um^-1\n"
                f"layer = {data['which_layer']}  index={data['layer_index']}\n"
                f"z offset = {data['z_offset_um']:.4g} um\n\n"
                f"R = {data['R']:.6g}\nT = {data['T']:.6g}\nA = {data['A']:.6g}\n\n"
                "Arrays computed internally:\n"
                "Ex,Ey,Ez,Hx,Hy,Hz are complex on the real-space grid.",
                va="top", ha="left", transform=ax_info.transAxes,
            )

            self.fig.tight_layout()
            self.canvas.draw_idle()
            self.status.setText("Field preview done. This was one energy/k-point only; full scan data were not changed.")
        except Exception as e:
            self.status.setText("Field preview error")
            QMessageBox.critical(self, "Field preview error", traceback.format_exc())

    def run_simulation(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Running", "A simulation is already running.")
            return
        try:
            params = self.get_params()
        except Exception as e:
            QMessageBox.critical(self, "Input error", str(e))
            return

        npol = len(pol_list_from_mode(params.get("pol", "p")))
        total = npol * params["E_points"] * params["kx_points"] * params["ky_points"]
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText(f"Running patterned + reference calculations... 0/{total} points")
        self.result = None

        self.worker = Worker(params)
        self.progress.setValue(0)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.cancelled.connect(self.on_cancelled)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def stop_simulation(self):
        if self.worker is not None and self.worker.isRunning():
            self.status.setText("Stopping after the current RCWA point finishes...")
            self.stop_btn.setEnabled(False)
            self.worker.stop()
        else:
            self.status.setText("No simulation is running.")

    def on_cancelled(self, partial_result):
        self.result = partial_result
        completed = partial_result.get("completed_points", 0)
        total = partial_result.get("total_points", 0)
        self.status.setText(f"Stopped by user after {completed}/{total} points")
        self.stop_btn.setEnabled(False)
        QMessageBox.information(
            self,
            "Stopped",
            f"Simulation stopped after {completed}/{total} points. "
            "Partial data are kept in memory and can be saved, but the plots are not refreshed "
            "to avoid showing incomplete zero-filled regions."
        )

    @staticmethod
    def _format_duration(seconds):
        try:
            seconds = float(seconds)
        except Exception:
            return "--"
        if not np.isfinite(seconds) or seconds < 0:
            return "--"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:d}h {m:02d}m {s:02d}s"
        if m > 0:
            return f"{m:d}m {s:02d}s"
        return f"{s:d}s"

    def on_progress(self, count, total, pct, ipol, npol, pol, E, kx, ky, elapsed, eta):
        self.progress.setMaximum(100)
        self.progress.setValue(pct)
        elapsed_s = self._format_duration(elapsed)
        eta_s = self._format_duration(eta)
        if str(pol).startswith("parallel") or not np.isfinite(kx) or not np.isfinite(ky):
            self.status.setText(
                f"{count}/{total} ({pct}%): energy blocks {ipol}/{npol}, {pol}, "
                f"last E={E:.4f} eV | elapsed {elapsed_s}, remaining ~{eta_s}"
            )
        else:
            self.status.setText(
                f"{count}/{total} ({pct}%): polarization {ipol}/{npol} ({pol}), "
                f"E={E:.4f} eV, kx={kx:.3f}, ky={ky:.3f} | "
                f"elapsed {elapsed_s}, remaining ~{eta_s}"
            )

    def on_done(self, result):
        self.result = result
        self.status.setText("Done; plotting results...")
        self.stop_btn.setEnabled(False)
        try:
            self.plot_results()
            self.status.setText("Done")
        except Exception:
            self.status.setText("Done, but plotting failed")
            QMessageBox.critical(self, "Plotting error", traceback.format_exc())

    def on_error(self, err):
        self.status.setText("Error")
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Simulation error", err)

    def compute_derivative_map(self, Z, energies, kxs, mode):
        """
        Return a derivative/edge-enhanced map with the same shape as Z.

        Z is shaped [energy_index, kx_index].
        np.gradient handles nonuniform axes if energies/kxs are arrays.
        """
        Z = np.asarray(Z, dtype=float)
        if Z.shape[0] < 2 or Z.shape[1] < 2:
            return np.full_like(Z, np.nan), "not enough points"

        dZ_dE, dZ_dk = np.gradient(Z, energies, kxs, edge_order=1)

        if mode.startswith("dR/dE") or mode.startswith("d/dE"):
            return dZ_dE, mode
        if mode.startswith("-dR/dE") or mode.startswith("-d/dE"):
            return -dZ_dE, mode
        if mode.startswith("|grad"):
            e_span = max(float(np.nanmax(energies) - np.nanmin(energies)), 1e-15)
            k_span = max(float(np.nanmax(kxs) - np.nanmin(kxs)), 1e-15)
            scale = k_span / e_span
            return np.sqrt(dZ_dE**2 + (scale * dZ_dk)**2), mode

        return dZ_dE, mode

    def plot_results(self):
        """
        Lightweight solver preview only.

        The solver now exports the complex p/s data for the viewer.  Here we only
        show geometry plus simple R maps for p and s with robust autoscaling.
        """
        if self.result is None:
            return

        self.fig.clear()
        ax_geom = self.fig.add_subplot(1, 3, 1)
        ax_p = self.fig.add_subplot(1, 3, 2)
        ax_s = self.fig.add_subplot(1, 3, 3)

        E = np.asarray(self.result["energies"], dtype=float)
        kx = np.asarray(self.result["kxs"], dtype=float)
        ky = np.asarray(self.result["kys"], dtype=float)
        ky_index = len(ky)//2 if len(ky) > 1 else 0
        extent = [kx[0], kx[-1], E[0], E[-1]]
        pol_labels = [str(x).lower().strip() for x in self.result.get("pol_labels", [])]

        def robust_limits(Z):
            Z = np.asarray(Z, dtype=float)
            finite = Z[np.isfinite(Z)]
            if finite.size == 0:
                return 0.0, 1.0
            lo, hi = np.nanpercentile(finite, [2, 98])
            if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi-lo) < 1e-15:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            if abs(hi-lo) < 1e-15:
                hi = lo + 1e-9
            return float(lo), float(hi)

        def get_R(label):
            lab = str(label).lower().strip()
            if lab in pol_labels:
                return np.asarray(self.result["R"][pol_labels.index(lab), :, :, ky_index], dtype=float)
            return np.full((len(E), len(kx)), np.nan, dtype=float)

        # Geometry preview. IMPORTANT: use the same geometry source as the solver.
        # If a Designer cell JSON was loaded, make_pattern_eps() rasterizes that cell;
        # otherwise it falls back to the built-in slit macrocell.
        try:
            params = self.result.get("params", {})
            eps_metal = gold_epsilon(params, float(E[len(E)//2])) if params else -10+1j
            eps_grid, Lx, Ly = make_pattern_eps(params, eps_metal, params.get("n_air", 1.0)**2)
            plot_lattice_array(ax_geom, np.real(eps_grid), params)
            if params.get("cell_design"):
                n_shapes = len(params.get("cell_design", {}).get("shapes", []))
                title = f"Geometry: designer cell ({n_shapes} shape{'s' if n_shapes != 1 else ''})"
            else:
                title = "Geometry: built-in slit macrocell"
            ax_geom.set_title(title)
            ax_geom.set_xlabel("x (um)")
            ax_geom.set_ylabel("y (um)")
        except Exception as exc:
            ax_geom.axis("off")
            ax_geom.set_title(f"Geometry preview failed\n{exc}")

        Rp = get_R("p")
        Rs = get_R("s")
        for ax, Z, title in [(ax_p, Rp, "p excitation: R"), (ax_s, Rs, "s excitation: R")]:
            vmin, vmax = robust_limits(Z)
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("kx (um$^{-1}$)")
            ax.set_ylabel("Energy (eV)")
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        self.fig.suptitle("Solver preview only — detailed polarization analysis in viewer", fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas.draw_idle()

    def save_npz(self):
        if self.result is None:
            QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save self-describing solver NPZ",
            "rcwa_solver_fields_v1_8.npz",
            "NumPy archive (*.npz)"
        )
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"

        params_for_save = dict(self.result["params"])
        params_for_save["top_layers"] = [l.to_dict() for l in params_for_save.get("top_layers", [])]
        if params_for_save.get("gold_material") is not None:
            params_for_save["gold_material"] = params_for_save["gold_material"].to_dict()

        config = self.current_config_dict()
        config["result_file_type"] = "RCWA_solver_self_describing_npz"
        config["solver_version"] = SOLVER_VERSION
        config["config_version"] = CONFIG_VERSION
        config["npz_schema_version"] = "1.1"
        config["reference_definition"] = {
            "reference_mode": params_for_save.get("reference_mode", "uniform_gold"),
            "uniform_gold": "selected top layers with use_in_reference=True + continuous Au film + glass",
            "patterned_gold": "selected top layers with use_in_reference=True + same patterned Au slit layer + glass",
            "no_gold": "selected top layers with use_in_reference=True + glass, no Au",
        }
        config["units"] = {
            "length": "micron",
            "energy": "eV",
            "kx_ky": "um^-1",
            "thickness": "micron",
            "angles": "degrees for input rotation_step; radians internally",
        }
        config["saved_arrays"] = {
            "pol_labels": "polarization basis calculations included",
            "energies": "photon energies, eV",
            "kxs": "in-plane wavevector x, um^-1",
            "kys": "in-plane wavevector y, um^-1",
            "R,T,A": "patterned stack total powers, shape [pol,E,kx,ky]",
            "Rref,Tref,Aref": "reference stack total powers, shape [pol,E,kx,ky]",
            "dRoverR": "(R - Rref)/Rref, shape [pol,E,kx,ky]",
            "b0,b0_ref": "complex reflected modal amplitudes if available",
            "aN,aN_ref": "complex transmitted modal amplitudes if available",
            "E_top,H_top,E_bottom,H_bottom": "complex Fourier fields saved for optional output polarization analysis",
            "G_orders": "diffraction order list from grcwa",
            "metadata_json": "full self-describing JSON metadata",
            "params_json": "numerical parameter dictionary as JSON, including Au material mode/file mapping",
            "R_RCP_all/T_RCP_all/A_RCP_all and LCP equivalents": "precomputed synthetic circular incident total flux maps for all propagating orders, shape [E,kx,ky]",
            "R_RCP_zeroth/T_RCP_zeroth/A_RCP_zeroth and LCP equivalents": "precomputed synthetic circular incident zeroth-order flux maps, shape [E,kx,ky]",
            "*_ref": "same precomputed maps for the reference region",
            "R_CD_all/R_CD_zeroth etc": "precomputed circular contrast maps (RCP-LCP)/(RCP+LCP)",
        }

        metadata_json = json.dumps(config, indent=2)
        params_json = json.dumps(params_for_save, indent=2)
        precomputed_common = precompute_common_maps(self.result)

        np.savez_compressed(
            path,
            metadata_json=np.array(metadata_json),
            params_json=np.array(params_json),
            pol_labels=self.result.get("pol_labels"),
            energies=self.result["energies"],
            kxs=self.result["kxs"],
            kys=self.result["kys"],
            R=self.result["R"],
            T=self.result["T"],
            A=self.result["A"],
            Rref=self.result["Rref"],
            Tref=self.result["Tref"],
            Aref=self.result["Aref"],
            dRoverR=self.result["dRoverR"],
            R_b_spec=self.result.get("R_b_spec"),
            phase_b_spec=self.result.get("phase_b_spec"),
            Ex_b_spec=self.result.get("Ex_b_spec"),
            Ey_b_spec=self.result.get("Ey_b_spec"),
            Ez_b_spec=self.result.get("Ez_b_spec"),
            E_RCP_b_spec=self.result.get("E_RCP_b_spec"),
            E_LCP_b_spec=self.result.get("E_LCP_b_spec"),
            b0=self.result.get("b0"),
            b0_ref=self.result.get("b0_ref"),
            aN=self.result.get("aN"),
            aN_ref=self.result.get("aN_ref"),
            E_top=self.result.get("E_top"),
            H_top=self.result.get("H_top"),
            E_bottom=self.result.get("E_bottom"),
            H_bottom=self.result.get("H_bottom"),
            E_top_ref=self.result.get("E_top_ref"),
            H_top_ref=self.result.get("H_top_ref"),
            E_bottom_ref=self.result.get("E_bottom_ref"),
            H_bottom_ref=self.result.get("H_bottom_ref"),
            G_orders=self.result.get("G_orders"),
            params=np.array([params_for_save], dtype=object),
            config_json=np.array(metadata_json),
            **precomputed_common,
        )

        QMessageBox.information(
            self,
            "Saved",
            f"Saved self-describing solver data to:\\n{path}\\n\\n"
            "No separate JSON file was written. The NPZ includes metadata_json and params_json."
        )

    def save_csv(self):
        if self.result is None:
            QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save quick-check CSV", "rcwa_quickcheck.csv", "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        E = self.result["energies"]
        kx = self.result["kxs"]
        ky = self.result["kys"]
        pol_labels = list(self.result.get("pol_labels", ["p"]))

        layer_summary = "; ".join([
            f"{l.name}:{l.thickness_um:g}um:ref={'yes' if l.use_in_reference else 'no'}:{self.material_summary(l.material)}"
            for l in self.result["params"].get("top_layers", [])
        ])
        ref_mode = self.result["params"].get("reference_mode", "uniform_gold")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_type", "RCWA_solver_quickcheck_csv"])
            writer.writerow(["reference_mode", ref_mode])
            writer.writerow(["top_layers", layer_summary])
            writer.writerow([
                "polarization", "energy_eV", "kx_um^-1", "ky_um^-1",
                "R", "T", "A", "R_reference", "T_reference", "A_reference", "DeltaR_over_R",
            ])
            for ipol, pol in enumerate(pol_labels):
                for i, energy in enumerate(E):
                    for j, kxv in enumerate(kx):
                        for k, kyv in enumerate(ky):
                            writer.writerow([
                                pol, energy, kxv, kyv,
                                self.result["R"][ipol, i, j, k],
                                self.result["T"][ipol, i, j, k],
                                self.result["A"][ipol, i, j, k],
                                self.result["Rref"][ipol, i, j, k],
                                self.result["Tref"][ipol, i, j, k],
                                self.result["Aref"][ipol, i, j, k],
                                self.result["dRoverR"][ipol, i, j, k],
                            ])

        QMessageBox.information(self, "Saved", f"Saved quick-check CSV to:\\n{path}")




def main():
    multiprocessing.freeze_support()
    _set_single_thread_blas_env()
    app = QApplication(sys.argv)
    win = RCWAQtGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
