#!/usr/bin/env python3
"""
LighTTice Viewer

Circular-polarization viewer using p and s solver runs and saved complex reflected
modal amplitudes b0.

This version deliberately avoids reconstructing CP by subtracting incident fields from
E_top/H_top. It uses b0 directly, reconstructs reflected E/H order-by-order in the
uniform superstrate, builds the complex Jones matrix, and only then squares to get
RR/LL/CD.

Required NPZ arrays:
    pol_labels containing "p" and "s"
    b0
    G_orders
    energies, kxs, kys

Circular reconstruction:
    1. Extract reflected modal amplitudes b0 for p and s incidence.
    2. Convert b0 to reflected E/H Fourier plane-wave fields in the top medium.
    3. Project each reflected diffraction order onto its local reflected s/p basis.
    4. Build the linear complex Jones matrix J_sp for each order.
    5. Transform to circular basis:
            R = (i*s + p)/sqrt(2)
            L = (-i*s + p)/sqrt(2)
    6. Plot RR, LL, CD, and DoCP.

The quantity can be computed for:
    - zeroth order only
    - sum over all propagating reflected orders
"""

import sys
import json
import csv
import re
import os
import tempfile
import shutil
import time
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox, QFileDialog,
    QMessageBox, QTextEdit, QGroupBox, QCheckBox, QListWidget, QProgressBar,
    QScrollArea, QGridLayout, QTabWidget, QSizePolicy
)

VIEWER_VERSION = "5.7-resizable-compact-layout"
HC_EV_UM = 1.239841984


def read_json(data, key):
    if key not in data.files:
        return {}
    try:
        return json.loads(str(data[key].tolist()))
    except Exception:
        try:
            return json.loads(str(data[key]))
        except Exception:
            return {}


def safe_contrast(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = a + b
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(den) & (np.abs(den) > 1e-15)
    out[m] = (a[m] - b[m]) / den[m]
    return out


def safe_divide(a, b, eps=1e-12):
    """Elementwise safe division for custom Python expressions.

    Returns NaN where the denominator is zero, too small, or non-finite.
    Use in the expression box as div(v1-v5, v2-v4).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype=float)
    aa = np.broadcast_to(a, out.shape)
    bb = np.broadcast_to(b, out.shape)
    m = np.isfinite(aa) & np.isfinite(bb) & (np.abs(bb) > eps)
    out[m] = aa[m] / bb[m]
    return out


def sanitize_map(Z, max_abs=1e100):
    """Make a plotted map safe for matplotlib without hiding real NaN masks."""
    Z = np.asarray(Z, dtype=float).copy()
    Z[~np.isfinite(Z)] = np.nan
    Z[np.abs(Z) > max_abs] = np.nan
    return Z


def robust_limits(Z, symmetric=False, positive=False, ignore_zero=False, p_low=2, p_high=98,
                  high_contrast=False):
    """Robust color limits for maps.

    For positive bounded quantities such as R/T/A, exact zero padding from
    non-propagating orders is ignored when estimating the limits.

    high_contrast=True is meant for raw R/T/A maps that are almost saturated
    near 1.  It clips the lower tail more aggressively so small resonance
    variations in the bright central region are visible instead of being
    washed out by dark edge/order-cutoff pixels.
    """
    finite = np.asarray(Z, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1, 1) if symmetric else (0, 1)

    if symmetric:
        vmax = np.nanpercentile(np.abs(finite), p_high)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return -float(vmax), float(vmax)

    vals = finite
    if ignore_zero:
        nz = finite[np.abs(finite) > 1e-12]
        if nz.size >= max(8, int(0.05 * finite.size)):
            vals = nz

    if high_contrast and vals.size > 20:
        # If the map has a bright plateau (typical raw R/T), do not let the
        # low-reflectance edge/cutoff pixels dominate the scale.
        med = np.nanmedian(vals)
        if positive and np.isfinite(med) and med > 0.45:
            p_low = max(p_low, 20)
            p_high = max(p_high, 99)

    lo = np.nanpercentile(vals, p_low)
    hi = np.nanpercentile(vals, p_high)
    if positive:
        lo = max(0.0, lo)
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
        if positive:
            lo = max(0.0, lo)
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            center = float(lo) if np.isfinite(lo) else 0.0
            span = max(abs(center), 1.0)
            lo = center - 0.05 * span
            hi = center + 0.10 * span
    return float(lo), float(hi)



def make_kp_matrix_uniform(omega, epsilon, kx, ky):
    """Same uniform-layer kp matrix as grcwa MakeKPMatrix(..., layer_type=0)."""
    kx = np.asarray(kx, dtype=complex)
    ky = np.asarray(ky, dtype=complex)
    nG = len(kx)
    Jk = np.vstack((-np.diag(ky), np.diag(kx)))
    JkkJT = Jk @ Jk.T
    return omega**2 * np.eye(2*nG, dtype=complex) - (1.0/epsilon) * JkkJT


def get_z_poynting_flux_np(ai, bi, omega, kp, phi, q, byorder=0):
    """Numpy copy of grcwa GetZPoyntingFlux.

    Returns forward, backward flux in Victor/grcwa normalization before multiplying
    by obj.normalization.  For reflected power, R = real(-backward)*normalization.
    """
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
    if byorder == 0:
        forward = np.sum(forward)
        backward = np.sum(backward)
    return forward, backward


def derivative_map(Z, energies, kxs, mode):
    Z = np.asarray(Z, dtype=float)
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return np.full_like(Z, np.nan)
    dE, dk = np.gradient(Z, energies, kxs, edge_order=1)
    if mode == "d/dE":
        return dE
    if mode == "-d/dE":
        return -dE
    e_span = max(float(np.nanmax(energies) - np.nanmin(energies)), 1e-15)
    k_span = max(float(np.nanmax(kxs) - np.nanmin(kxs)), 1e-15)
    return np.sqrt(dE**2 + ((k_span / e_span) * dk)**2)



class CommentToggleTextEdit(QTextEdit):
    """QTextEdit with Ctrl+/ line-comment toggle for the custom Python box."""

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Slash and (event.modifiers() & Qt.ControlModifier):
            self.toggle_line_comments()
            event.accept()
            return
        super().keyPressEvent(event)

    def toggle_line_comments(self):
        cursor = self.textCursor()
        text = self.toPlainText()
        if not text:
            return

        sel_start = cursor.selectionStart()
        sel_end = cursor.selectionEnd()

        # If the selection ends exactly at the start of a line, do not include
        # that following line. This matches normal editor behavior better.
        effective_end = sel_end
        if sel_end > sel_start and sel_end <= len(text) and text[sel_end - 1:sel_end] == "\n":
            effective_end = sel_end - 1

        line_start = text.rfind("\n", 0, sel_start) + 1
        next_newline = text.find("\n", effective_end)
        line_end = len(text) if next_newline == -1 else next_newline

        block = text[line_start:line_end]
        lines = block.split("\n")

        def is_commented(line):
            stripped = line.lstrip()
            return stripped.startswith("#") or stripped == ""

        nonblank = [ln for ln in lines if ln.strip()]
        uncomment_mode = bool(nonblank) and all(is_commented(ln) for ln in nonblank)

        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue

            indent_len = len(line) - len(line.lstrip())
            indent = line[:indent_len]
            body = line[indent_len:]

            if uncomment_mode:
                if body.startswith("# "):
                    body = body[2:]
                elif body.startswith("#"):
                    body = body[1:]
                new_lines.append(indent + body)
            else:
                new_lines.append(indent + "# " + body)

        new_block = "\n".join(new_lines)

        edit_cursor = self.textCursor()
        edit_cursor.beginEditBlock()
        edit_cursor.setPosition(line_start)
        edit_cursor.setPosition(line_end, QTextCursor.KeepAnchor)
        edit_cursor.insertText(new_block)
        edit_cursor.setPosition(line_start)
        edit_cursor.setPosition(line_start + len(new_block), QTextCursor.KeepAnchor)
        edit_cursor.endEditBlock()
        self.setTextCursor(edit_cursor)


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LighTTice Viewer v{VIEWER_VERSION} — polarization analysis")
        self.resize(1450, 880)
        self.data = None
        self.path = None
        self.meta = {}
        self.current_map = None
        self.current_quantity = None
        self._cache = {}
        self.custom_vars = {}
        self.custom_var_specs = []
        self.custom_var_counter = 0
        self._busy_depth = 0
        self._plotting = False
        self.build_ui()

    def build_ui(self):
        """Compact 24-inch monitor layout.

        The viewer has grown into an analysis workbench, so the UI is organized as:
        1) a compact control band at the top,
        2) advanced tools in tabs directly below,
        3) a large Matplotlib canvas that receives all remaining space.

        This keeps the useful controls visible without stealing space from the plot.
        """
        main = QWidget()
        self.setCentralWidget(main)
        root = QVBoxLayout(main)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ------------------------------------------------------------------
        # RESIZABLE WORKSPACE: compact controls on top, large plot below
        # ------------------------------------------------------------------
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        # ------------------------------------------------------------------
        # TOP CONTROL BAND
        # ------------------------------------------------------------------
        control_band = QWidget()
        control_band.setMinimumHeight(235)
        control_band.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_band)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        self.main_splitter.addWidget(control_band)

        # Row 1: file/status + main analysis + display controls
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        control_layout.addLayout(row1)

        # ----- File/status group
        file_box = QGroupBox("File / status")
        file_box.setMinimumWidth(360)
        file_layout = QGridLayout(file_box)
        file_layout.setContentsMargins(8, 6, 8, 6)
        file_layout.setHorizontalSpacing(6)
        file_layout.setVerticalSpacing(4)
        row1.addWidget(file_box, 1)

        load_btn = QPushButton("Load NPZ")
        load_btn.clicked.connect(self.load_npz)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.reload_npz)
        file_layout.addWidget(load_btn, 0, 0)
        file_layout.addWidget(reload_btn, 0, 1)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(False)
        self.file_label.setToolTip("Loaded NPZ file")
        file_layout.addWidget(self.file_label, 1, 0, 1, 4)

        self.loading_label = QLabel("Ready")
        self.loading_label.setWordWrap(False)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setVisible(False)
        self.loading_bar.setMaximumHeight(14)
        file_layout.addWidget(QLabel("Status:"), 2, 0)
        file_layout.addWidget(self.loading_label, 2, 1, 1, 2)
        file_layout.addWidget(self.loading_bar, 2, 3)

        # ----- Analysis group
        analysis_box = QGroupBox("Analysis")
        analysis_layout = QGridLayout(analysis_box)
        analysis_layout.setContentsMargins(8, 6, 8, 6)
        analysis_layout.setHorizontalSpacing(6)
        analysis_layout.setVerticalSpacing(4)
        row1.addWidget(analysis_box, 2)

        self.region_box = QComboBox()
        self.region_box.addItems(["Sample", "Reference"])
        self.region_box.setToolTip("Choose whether maps are computed from sample data or from the reference stack. Delta quantities still compare Sample - Reference.")
        self.pol_box = QComboBox()
        self.pol_box.addItems(["s", "p", "RCP", "LCP", "Custom Jones"])
        self.pol_box.setToolTip("Incident polarization synthesized from saved p and s complex modal data.")
        self.quantity_box = QComboBox()
        self.quantity_box.addItems([
            "R",
            "R (RT_Solve saved)",
            "R (b0 reconstructed)",
            "1 - R",
            "T",
            "T (RT_Solve saved)",
            "T (aN reconstructed)",
            "A",
            "DeltaR",
            "DeltaR_over_R",
            "d/dE of R",
            "-d/dE of R",
            "|grad R|",
            "CD",
            "DoCP",
        ])
        self.quantity_box.setToolTip("R and T use RCWA Poynting-flux reconstruction from b0/aN for Analyzer=None. Analyzer channels use complex far-field projections. A = 1 - R - T.")
        self.analyzer_box = QComboBox()
        self.analyzer_box.addItems(["None", "s", "p", "RCP", "LCP", "Custom Jones"])
        self.analyzer_box.setToolTip("Output analyzer. None means total power. For R it analyzes reflected field; for T it analyzes transmitted field.")
        self.order_mode_box = QComboBox()
        self.order_mode_box.addItems(["zeroth order", "all propagating reflected orders"])
        self.ky_box = QComboBox()

        labels_widgets = [
            ("Region", self.region_box),
            ("Incident", self.pol_box),
            ("Quantity", self.quantity_box),
            ("Analyzer", self.analyzer_box),
            ("Order", self.order_mode_box),
            ("ky", self.ky_box),
        ]
        for col, (lab, widget) in enumerate(labels_widgets):
            analysis_layout.addWidget(QLabel(lab), 0, col)
            widget.setMinimumWidth(105)
            analysis_layout.addWidget(widget, 1, col)
        self.quantity_box.currentTextChanged.connect(self.update_analyzer_enabled)

        # ----- Display group
        display_box = QGroupBox("Display")
        display_layout = QGridLayout(display_box)
        display_layout.setContentsMargins(8, 6, 8, 6)
        display_layout.setHorizontalSpacing(6)
        display_layout.setVerticalSpacing(4)
        row1.addWidget(display_box, 2)

        self.plot_mode_box = QComboBox()
        self.plot_mode_box.addItems([
            "2D map",
            "E cut: value vs kx",
            "kx cut: value vs E",
            "2D map + both cuts",
        ])
        self.cut_E_edit = QLineEdit("")
        self.cut_E_edit.setPlaceholderText("auto")
        self.cut_kx_edit = QLineEdit("")
        self.cut_kx_edit.setPlaceholderText("auto")
        self.vmin_edit = QLineEdit("")
        self.vmin_edit.setPlaceholderText("auto")
        self.vmax_edit = QLineEdit("")
        self.vmax_edit.setPlaceholderText("auto")
        self.auto_scale_check = QCheckBox("Robust scale")
        self.auto_scale_check.setChecked(True)
        self.show_dashed_lines_check = QCheckBox("Guide lines")
        self.show_dashed_lines_check.setChecked(True)
        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self.plot)

        display_layout.addWidget(QLabel("Plot mode"), 0, 0)
        display_layout.addWidget(self.plot_mode_box, 1, 0, 1, 2)
        display_layout.addWidget(QLabel("Cut E"), 0, 2)
        display_layout.addWidget(self.cut_E_edit, 1, 2)
        display_layout.addWidget(QLabel("Cut kx"), 0, 3)
        display_layout.addWidget(self.cut_kx_edit, 1, 3)
        display_layout.addWidget(QLabel("vmin"), 2, 0)
        display_layout.addWidget(self.vmin_edit, 3, 0)
        display_layout.addWidget(QLabel("vmax"), 2, 1)
        display_layout.addWidget(self.vmax_edit, 3, 1)
        display_layout.addWidget(self.auto_scale_check, 3, 2)
        display_layout.addWidget(self.show_dashed_lines_check, 3, 3)
        display_layout.addWidget(plot_btn, 2, 2, 1, 2)

        # ------------------------------------------------------------------
        # ADVANCED / SCRIPT BAND
        # ------------------------------------------------------------------
        # A horizontal band: the Python expression editor gets most of the width.
        # Custom Jones and export/config live in a compact tab stack on the right.
        # The whole top band is vertically resizable via the splitter above.
        advanced_row = QHBoxLayout()
        advanced_row.setSpacing(8)
        control_layout.addLayout(advanced_row, 1)

        # Python expression group
        python_box = QGroupBox("Python expression")
        python_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        python_layout = QGridLayout(python_box)
        python_layout.setContentsMargins(8, 6, 8, 6)
        python_layout.setHorizontalSpacing(8)
        python_layout.setVerticalSpacing(4)
        advanced_row.addWidget(python_box, 4)

        assign_btn = QPushButton("Assign current")
        assign_btn.clicked.connect(self.assign_current_signal)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_custom_signals)
        python_layout.addWidget(assign_btn, 0, 0)
        python_layout.addWidget(clear_btn, 0, 1)

        py_help = QLabel("Use v1, v2... or S(...). For ratios use div(num, den). Ctrl+/ toggles comments.")
        py_help.setWordWrap(False)
        python_layout.addWidget(py_help, 0, 2, 1, 3)

        self.custom_signal_list = QListWidget()
        self.custom_signal_list.setMinimumWidth(390)
        self.custom_signal_list.setMinimumHeight(92)
        self.custom_signal_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        python_layout.addWidget(self.custom_signal_list, 1, 0, 2, 2)

        self.custom_expr_edit = CommentToggleTextEdit()
        self.custom_expr_edit.setMinimumHeight(92)
        self.custom_expr_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.custom_expr_edit.setPlaceholderText(
            "Examples:\n"
            "v1 - v2\n"
            "# v3 - v4\n"
            "S('Custom Jones','R','Custom Jones','zeroth','Sample') - S('s','T','RCP','all','Reference')\n"
            "(s,R,RCP,zeroth,Sample) - (s,T,RCP,all_order,Reference) / (RCP,A,None,zeroth,Sample)"
        )
        python_layout.addWidget(self.custom_expr_edit, 1, 2, 1, 3)

        custom_plot_btn = QPushButton("Plot Python expression")
        custom_plot_btn.clicked.connect(self.plot_custom_expression)
        python_layout.addWidget(custom_plot_btn, 2, 2, 1, 3)
        python_layout.setColumnStretch(0, 0)
        python_layout.setColumnStretch(1, 0)
        python_layout.setColumnStretch(2, 2)
        python_layout.setColumnStretch(3, 2)
        python_layout.setColumnStretch(4, 2)
        python_layout.setRowStretch(1, 1)

        # Right-side compact tabs for less frequently used tools
        side_tabs = QTabWidget()
        side_tabs.setMinimumWidth(430)
        side_tabs.setMaximumWidth(520)
        side_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        advanced_row.addWidget(side_tabs, 1)

        # Custom Jones tab
        jones_tab = QWidget()
        jones_layout = QGridLayout(jones_tab)
        jones_layout.setContentsMargins(8, 6, 8, 6)
        jones_layout.setHorizontalSpacing(8)
        jones_layout.setVerticalSpacing(4)
        side_tabs.addTab(jones_tab, "Custom Jones")
        self.inc_custom_ratio_edit = QLineEdit("1.0")
        self.inc_custom_ratio_edit.setToolTip("Incident custom Jones amplitude ratio |Es/Ep|. RCP/LCP-like states use ratio=1.")
        self.inc_custom_phase_edit = QLineEdit("90.0")
        self.inc_custom_phase_edit.setToolTip("Incident custom Jones phase arg(Es/Ep) in degrees. +90 is RCP-like in the viewer convention.")
        self.ana_custom_ratio_edit = QLineEdit("1.0")
        self.ana_custom_ratio_edit.setToolTip("Analyzer custom Jones amplitude ratio |Es/Ep|. RCP/LCP-like analyzers use ratio=1.")
        self.ana_custom_phase_edit = QLineEdit("90.0")
        self.ana_custom_phase_edit.setToolTip("Analyzer custom Jones phase arg(Es/Ep) in degrees. +90 is RCP-like in the viewer convention.")
        jones_layout.addWidget(QLabel("Inc |Es/Ep|"), 0, 0)
        jones_layout.addWidget(self.inc_custom_ratio_edit, 0, 1)
        jones_layout.addWidget(QLabel("Inc φ (deg)"), 0, 2)
        jones_layout.addWidget(self.inc_custom_phase_edit, 0, 3)
        jones_layout.addWidget(QLabel("Ana |Es/Ep|"), 1, 0)
        jones_layout.addWidget(self.ana_custom_ratio_edit, 1, 1)
        jones_layout.addWidget(QLabel("Ana φ (deg)"), 1, 2)
        jones_layout.addWidget(self.ana_custom_phase_edit, 1, 3)
        note_jones = QLabel("Custom Jones: [s,p]=[r exp(iφ), 1], normalized.")
        note_jones.setWordWrap(True)
        jones_layout.addWidget(note_jones, 2, 0, 1, 4)
        jones_layout.setColumnStretch(4, 1)

        # Export/config tab
        export_tab = QWidget()
        export_layout = QGridLayout(export_tab)
        export_layout.setContentsMargins(8, 6, 8, 6)
        export_layout.setHorizontalSpacing(6)
        export_layout.setVerticalSpacing(6)
        side_tabs.addTab(export_tab, "Export / config")

        csv_btn = QPushButton("Export CSV")
        csv_btn.clicked.connect(self.export_csv)
        png_btn = QPushButton("Save PNG")
        png_btn.clicked.connect(self.export_png)
        config_btn = QPushButton("Export solver JSON")
        config_btn.setToolTip("Write the simulation configuration embedded in the loaded NPZ to JSON. Load it back into the solver with Load Config.")
        config_btn.clicked.connect(self.export_solver_config_json)
        save_viewer_cfg_btn = QPushButton("Save viewer config")
        save_viewer_cfg_btn.setToolTip("Save current viewer controls, cuts, custom expression, and assigned signal definitions to JSON.")
        save_viewer_cfg_btn.clicked.connect(self.save_viewer_config_json)
        load_viewer_cfg_btn = QPushButton("Load viewer config")
        load_viewer_cfg_btn.setToolTip("Restore viewer controls, cuts, custom expression, and assigned signal definitions from JSON.")
        load_viewer_cfg_btn.clicked.connect(self.load_viewer_config_json)
        save_cache_btn = QPushButton("Write maps to NPZ")
        save_cache_btn.setToolTip("Append/overwrite viewer-computed 2D maps in the loaded NPZ. A .bak backup is created first.")
        save_cache_btn.clicked.connect(self.write_computed_maps_to_npz)

        export_layout.addWidget(csv_btn, 0, 0)
        export_layout.addWidget(png_btn, 0, 1)
        export_layout.addWidget(config_btn, 0, 2)
        export_layout.addWidget(save_viewer_cfg_btn, 1, 0)
        export_layout.addWidget(load_viewer_cfg_btn, 1, 1)
        export_layout.addWidget(save_cache_btn, 1, 2)
        export_layout.setColumnStretch(3, 1)

        # Metadata tab
        meta_tab = QWidget()
        meta_layout = QVBoxLayout(meta_tab)
        meta_layout.setContentsMargins(6, 6, 6, 6)
        side_tabs.addTab(meta_tab, "Metadata")
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        meta_layout.addWidget(self.meta_text)

        # ------------------------------------------------------------------
        # PLOT AREA
        # ------------------------------------------------------------------
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        self.fig = Figure(figsize=(13.5, 7.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        self.main_splitter.addWidget(plot_container)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([300, 760])

        # Auto-plot on core choices. Avoid connecting free text boxes, otherwise editing
        # cut values or expressions can trigger expensive recomputation while typing.
        for w in [
            self.pol_box,
            self.quantity_box,
            self.analyzer_box,
            self.region_box,
            self.ky_box,
            self.order_mode_box,
            self.plot_mode_box,
        ]:
            try:
                w.currentIndexChanged.connect(self.plot)
            except Exception:
                pass

        for w in [self.auto_scale_check, self.show_dashed_lines_check]:
            try:
                w.stateChanged.connect(self.plot)
            except Exception:
                pass

    def begin_busy(self, message="Working..."):
        """Show an indeterminate progress indicator and keep the GUI responsive."""
        self._busy_depth += 1
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        except Exception:
            pass
        try:
            self.loading_label.setText(message)
            self.loading_bar.setVisible(True)
            self.statusBar().showMessage(message)
        except Exception:
            pass
        QApplication.processEvents()

    def end_busy(self, message="Ready"):
        self._busy_depth = max(0, self._busy_depth - 1)
        if self._busy_depth == 0:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            try:
                self.loading_bar.setVisible(False)
                self.loading_label.setText(message)
                self.statusBar().showMessage(message, 3000)
            except Exception:
                pass
            QApplication.processEvents()

    def load_npz(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load RCWA solver NPZ", "", "NumPy archive (*.npz)")
        if path:
            self.open_npz(path)

    def reload_npz(self):
        if self.path:
            self.open_npz(self.path)

    def open_npz(self, path):
        self.begin_busy("Loading NPZ...")
        try:
            self.data = np.load(path, allow_pickle=True)
            self.path = path
            self.meta = read_json(self.data, "metadata_json") or read_json(self.data, "config_json")
            self._cache = {}
            self.file_label.setText(path)
            self.populate_controls()
            self.update_metadata()
            # Do not auto-plot CP on load; user chooses quantity and presses Plot.
            self.end_busy("Loaded NPZ")
        except Exception as e:
            self.end_busy("Load failed")
            QMessageBox.critical(self, "Load error", str(e))

    def pol_labels(self):
        if self.data is None or "pol_labels" not in self.data.files:
            return np.array(["p"], dtype=object)
        return self.data["pol_labels"]

    def pol_index(self, label):
        labels = [str(x).lower().strip() for x in self.pol_labels()]
        try:
            return labels.index(str(label).lower().strip())
        except ValueError:
            return None

    def populate_controls(self):
        # Incident/analyzer choices are fixed analysis bases; the NPZ itself should contain p and s.
        self.pol_box.blockSignals(True)
        self.ky_box.blockSignals(True)
        current_exc = self.pol_box.currentText() if self.pol_box.count() else "s"
        self.pol_box.clear()
        self.pol_box.addItems(["s", "p", "RCP", "LCP", "Custom Jones"])
        idx = self.pol_box.findText(current_exc)
        self.pol_box.setCurrentIndex(max(0, idx))
        self.ky_box.clear()
        for i, ky in enumerate(self.data["kys"]):
            self.ky_box.addItem(f"{i}: {float(ky):.4g} um^-1")
        self.pol_box.blockSignals(False)
        self.ky_box.blockSignals(False)
        self.update_analyzer_enabled()

    def update_analyzer_enabled(self):
        """Enable analyzer for field-resolved R/T quantities. A/CD/DoCP use fixed definitions."""
        if not hasattr(self, "analyzer_box") or not hasattr(self, "quantity_box"):
            return
        q = self.quantity_box.currentText()
        enabled = q in [
            "R", "1 - R", "DeltaR", "DeltaR_over_R", "d/dE of R", "-d/dE of R", "|grad R",
            "T",
        ]
        self.analyzer_box.setEnabled(enabled)
        if not enabled:
            idx = self.analyzer_box.findText("None")
            if idx >= 0:
                self.analyzer_box.setCurrentIndex(idx)

    def input_float(self, key, default):
        try:
            return float(self.meta.get("inputs", {}).get(key, default))
        except Exception:
            return default

    def n_air(self):
        return self.input_float("n_air", 1.0)

    def n_glass(self):
        return self.input_float("n_glass", 1.5)

    def Lx_Ly(self):
        px = self.input_float("period_x", 1.0)
        py = self.input_float("period_y", 1.0)
        nx = self.input_float("n_cells_x", 1.0)
        ny = self.input_float("n_cells_y", 1.0)
        return px * nx, py * ny

    def zero_order_index(self):
        if "G_orders" not in self.data.files:
            return 0
        G = np.asarray(self.data["G_orders"])
        idx = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
        return int(idx[0]) if len(idx) else 0

    def order_geometry(self, E_eV, kx_inc, ky_inc):
        return self.side_order_geometry(E_eV, kx_inc, ky_inc, side="reflection")

    def side_order_geometry(self, E_eV, kx_inc, ky_inc, side="reflection"):
        G = np.asarray(self.data["G_orders"], dtype=float)
        Lx, Ly = self.Lx_Ly()
        kx = float(kx_inc) + 2*np.pi*G[:, 0]/Lx
        ky = float(ky_inc) + 2*np.pi*G[:, 1]/Ly

        k0 = 2*np.pi*float(E_eV)/HC_EV_UM
        nmed = self.n_air() if side == "reflection" else self.n_glass()
        nk0 = nmed * k0
        kpar2 = kx*kx + ky*ky
        prop = kpar2 <= nk0*nk0 + 1e-12
        kz_abs = np.zeros_like(kx)
        kz_abs[prop] = np.sqrt(np.maximum(nk0*nk0 - kpar2[prop], 0.0))
        kz = -kz_abs if side == "reflection" else kz_abs
        return kx, ky, kz, kz_abs, prop, nk0

    def incident_vectors(self, E_eV, kx, ky, pol):
        _, _, _, _, _, nk0 = self.order_geometry(E_eV, kx, ky)
        kt = np.hypot(float(kx), float(ky))
        if nk0 <= 0:
            theta = 0.0
        else:
            theta = np.arcsin(np.clip(kt/nk0, -1, 1))
        phi = np.arctan2(float(ky), float(kx)) if kt > 1e-15 else 0.0

        khat = np.array([
            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta),
        ], dtype=complex)

        e_s = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=complex)
        e_p = np.array([
            np.cos(theta)*np.cos(phi),
            np.cos(theta)*np.sin(phi),
            -np.sin(theta),
        ], dtype=complex)
        Evec = e_p if str(pol).lower().startswith("p") else e_s
        Hvec = np.cross(khat, Evec)
        return Evec, Hvec

    def sp_basis_for_direction(self, kx, ky, kz, nk0):
        kt = np.hypot(kx, ky)
        if kt < 1e-15:
            e_s = np.array([0.0, 1.0, 0.0], dtype=complex)
            e_p = np.array([1.0 if kz >= 0 else -1.0, 0.0, 0.0], dtype=complex)
        else:
            e_s = np.array([-ky/kt, kx/kt, 0.0], dtype=complex)
            khat = np.array([kx/nk0, ky/nk0, kz/nk0], dtype=complex)
            e_p = np.cross(e_s, khat)
            e_p = e_p / max(np.linalg.norm(e_p), 1e-30)
        return e_s, e_p

    def reflected_sp_basis(self, kx, ky, kz_ref, nk0):
        return self.sp_basis_for_direction(kx, ky, kz_ref, nk0)

    def transmitted_sp_basis(self, kx, ky, kz_tr, nk0):
        return self.sp_basis_for_direction(kx, ky, kz_tr, nk0)

    def reflected_fields_for_pol(self, pol, ky_index, b0_key="b0"):
        """
        Reconstruct reflected E/H Fourier fields from saved b0.

        b0 is preferred over E_top/H_top because b0 contains only the reflected
        outgoing amplitudes. Therefore there is no incident-field subtraction,
        no dependence on a sampled z plane, and fewer sign/normalization traps.

        Output shape:
            E, H = (nE, nkx, 3, nG)
        """
        ipol = self.pol_index(pol)
        if ipol is None:
            raise ValueError(f"Missing polarization {pol}; found {list(self.pol_labels())}")
        if b0_key not in self.data.files:
            raise ValueError(f"NPZ must contain {b0_key}. Re-run/export with the solver that saves complex b0 amplitudes.")

        b0_all = np.array(self.data[b0_key][ipol, :, :, ky_index, :], dtype=complex)
        energies = np.asarray(self.data["energies"], dtype=float)
        kxs_inc = np.asarray(self.data["kxs"], dtype=float)
        ky_inc = float(self.data["kys"][ky_index])

        nE, nkx, nAmp = b0_all.shape
        nG = nAmp // 2
        E = np.full((nE, nkx, 3, nG), np.nan + 1j*np.nan, dtype=complex)
        H = np.full((nE, nkx, 3, nG), np.nan + 1j*np.nan, dtype=complex)

        eps_top = self.n_air() ** 2

        for iE, Eev in enumerate(energies):
            omega = 2*np.pi*float(Eev)/HC_EV_UM
            for ikx, kx0 in enumerate(kxs_inc):
                kx, ky, kz_ref, kz_abs, prop, nk0 = self.order_geometry(Eev, kx0, ky_inc)
                b = b0_all[iE, ikx, :]
                if b.size < 2*nG:
                    continue

                Hx = b[:nG]
                Hy = b[nG:2*nG]

                # Reflected wavevector is (kx, ky, -q), with q = kz_abs >= 0.
                q = kz_abs.astype(complex)
                good = prop & (np.abs(q) > 1e-15) & np.isfinite(Hx) & np.isfinite(Hy)

                # H_z from k·H = 0: kx Hx + ky Hy - q Hz = 0.
                Hz = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                Hz[good] = (kx[good]*Hx[good] + ky[good]*Hy[good]) / q[good]

                # E = -(k × H)/(omega*epsilon) for exp(-i omega t), matching grcwa convention.
                Ex = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                Ey = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                Ez = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                kzr = -q
                Ex[good] = -((ky[good]*Hz[good] - kzr[good]*Hy[good]) / (omega*eps_top))
                Ey[good] = -((kzr[good]*Hx[good] - kx[good]*Hz[good]) / (omega*eps_top))
                Ez[good] = -((kx[good]*Hy[good] - ky[good]*Hx[good]) / (omega*eps_top))

                E[iE, ikx, 0, :] = Ex
                E[iE, ikx, 1, :] = Ey
                E[iE, ikx, 2, :] = Ez
                H[iE, ikx, 0, :] = Hx
                H[iE, ikx, 1, :] = Hy
                H[iE, ikx, 2, :] = Hz
        return E, H

    def transmitted_fields_for_pol(self, pol, ky_index, aN_key="aN"):
        """Reconstruct transmitted E/H Fourier fields from saved aN in the bottom medium."""
        ipol = self.pol_index(pol)
        if ipol is None:
            raise ValueError(f"Missing polarization {pol}; found {list(self.pol_labels())}")
        if aN_key not in self.data.files:
            raise ValueError(f"NPZ must contain {aN_key}. Re-run/export with solver v4.3+ that saves transmitted complex aN amplitudes.")

        aN_all = np.array(self.data[aN_key][ipol, :, :, ky_index, :], dtype=complex)
        energies = np.asarray(self.data["energies"], dtype=float)
        kxs_inc = np.asarray(self.data["kxs"], dtype=float)
        ky_inc = float(self.data["kys"][ky_index])

        nE, nkx, nAmp = aN_all.shape
        nG = nAmp // 2
        E = np.full((nE, nkx, 3, nG), np.nan + 1j*np.nan, dtype=complex)
        H = np.full((nE, nkx, 3, nG), np.nan + 1j*np.nan, dtype=complex)

        eps_bot = self.n_glass() ** 2

        for iE, Eev in enumerate(energies):
            omega = 2*np.pi*float(Eev)/HC_EV_UM
            for ikx, kx0 in enumerate(kxs_inc):
                kx, ky, kz_tr, kz_abs, prop, nk0 = self.side_order_geometry(Eev, kx0, ky_inc, side="transmission")
                a = aN_all[iE, ikx, :]
                if a.size < 2*nG:
                    continue

                Hx = a[:nG]
                Hy = a[nG:2*nG]
                q = kz_abs.astype(complex)
                good = prop & (np.abs(q) > 1e-15) & np.isfinite(Hx) & np.isfinite(Hy)

                # Forward transmitted wavevector is (kx, ky, +q). From k·H=0:
                Hz = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                Hz[good] = -(kx[good]*Hx[good] + ky[good]*Hy[good]) / q[good]

                # Same convention as reflection: E = -(k × H)/(omega*epsilon).
                Ex = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                Ey = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                Ez = np.full(nG, np.nan + 1j*np.nan, dtype=complex)
                kzt = q
                Ex[good] = -((ky[good]*Hz[good] - kzt[good]*Hy[good]) / (omega*eps_bot))
                Ey[good] = -((kzt[good]*Hx[good] - kx[good]*Hz[good]) / (omega*eps_bot))
                Ez[good] = -((kx[good]*Hy[good] - ky[good]*Hx[good]) / (omega*eps_bot))

                E[iE, ikx, 0, :] = Ex
                E[iE, ikx, 1, :] = Ey
                E[iE, ikx, 2, :] = Ez
                H[iE, ikx, 0, :] = Hx
                H[iE, ikx, 1, :] = Hy
                H[iE, ikx, 2, :] = Hz
        return E, H

    def sp_amplitudes_from_fields(self, Efield, Hfield, iE, ikx, ky_index, side="reflection"):
        """
        Project Fourier E field onto local s/p basis for each propagating order.
        side='reflection' gives reflected basis/power (-Sz); side='transmission' gives transmitted basis/power (+Sz).
        """
        Eev = float(self.data["energies"][iE])
        kx_inc = float(self.data["kxs"][ikx])
        ky_inc = float(self.data["kys"][ky_index])
        kx, ky, kz_dir, kz_abs, prop, nk0 = self.side_order_geometry(Eev, kx_inc, ky_inc, side=side)

        nG = len(kx)
        s_amp = np.zeros(nG, dtype=complex)
        p_amp = np.zeros(nG, dtype=complex)
        weights = np.zeros(nG, dtype=float)

        for m in range(nG):
            if not prop[m] or kz_abs[m] <= 1e-15:
                continue
            e_s, e_p = self.sp_basis_for_direction(kx[m], ky[m], kz_dir[m], nk0)
            Evec = Efield[iE, ikx, :, m]
            Hvec = Hfield[iE, ikx, :, m]
            if not np.all(np.isfinite(Evec)) or not np.all(np.isfinite(Hvec)):
                continue
            s_amp[m] = np.vdot(e_s, Evec)
            p_amp[m] = np.vdot(e_p, Evec)

            # Power flux. Reflection uses -Sz; transmission uses +Sz. Omit universal 1/2.
            Sz = np.real(Evec[0]*np.conjugate(Hvec[1]) - Evec[1]*np.conjugate(Hvec[0]))
            P = -Sz if side == "reflection" else Sz
            weights[m] = max(0.0, P) / max((abs(s_amp[m])**2 + abs(p_amp[m])**2), 1e-30)

        return s_amp, p_amp, weights, prop

    def has_direct_cp(self):
        labels = [str(x).lower().strip() for x in self.pol_labels()]
        return any(x.startswith("rcp") for x in labels) and any(x.startswith("lcp") for x in labels)

    def direct_cp_maps_for_ky(self, ky_index):
        """
        Direct circular workflow:
          - Solver runs RCP and LCP as independent incident polarizations.
          - CD is computed directly from their total reflected powers.
          - If b0 is available, reflected output is decomposed into circular
            channels: RCP→RCP, RCP→LCP, LCP→RCP, LCP→LCP.
        """
        mode = self.order_mode_box.currentText()
        key = ("direct_cp", ky_index, mode)
        if key in self._cache:
            return self._cache[key]

        ir = self.pol_index("RCP")
        il = self.pol_index("LCP")
        if ir is None or il is None:
            raise ValueError("Direct CP maps require pol_labels containing RCP and LCP. Re-run solver with Pol. basis = RCP and LCP.")

        R_RCP = np.asarray(self.data["R"][ir, :, :, ky_index], dtype=float)
        R_LCP = np.asarray(self.data["R"][il, :, :, ky_index], dtype=float)
        T_RCP = np.asarray(self.data["T"][ir, :, :, ky_index], dtype=float) if "T" in self.data.files else np.full_like(R_RCP, np.nan)
        T_LCP = np.asarray(self.data["T"][il, :, :, ky_index], dtype=float) if "T" in self.data.files else np.full_like(R_RCP, np.nan)
        if "A" in self.data.files:
            A_RCP = np.asarray(self.data["A"][ir, :, :, ky_index], dtype=float)
            A_LCP = np.asarray(self.data["A"][il, :, :, ky_index], dtype=float)
        else:
            A_RCP = 1.0 - R_RCP - T_RCP
            A_LCP = 1.0 - R_LCP - T_LCP
        CD = safe_contrast(R_RCP, R_LCP)

        RR_out = np.full_like(CD, np.nan, dtype=float)  # RCP input -> RCP output
        LR_out = np.full_like(CD, np.nan, dtype=float)  # RCP input -> LCP output
        RL_out = np.full_like(CD, np.nan, dtype=float)  # LCP input -> RCP output
        LL_out = np.full_like(CD, np.nan, dtype=float)  # LCP input -> LCP output
        DoCP_RCP_in = np.full_like(CD, np.nan, dtype=float)
        DoCP_LCP_in = np.full_like(CD, np.nan, dtype=float)

        if "b0" in self.data.files:
            try:
                energies = self.data["energies"]
                kxs = self.data["kxs"]
                oi = self.zero_order_index()
                # Basis vector order from sp_amplitudes_from_fields is [s, p].
                Rv = np.array([1j, 1.0], dtype=complex) / np.sqrt(2.0)
                Lv = np.array([-1j, 1.0], dtype=complex) / np.sqrt(2.0)

                def decompose_input(pol):
                    Efield, Hfield = self.reflected_fields_for_pol(pol, ky_index)
                    Rout = np.zeros_like(CD, dtype=float)
                    Lout = np.zeros_like(CD, dtype=float)
                    for i in range(len(energies)):
                        for j in range(len(kxs)):
                            s_amp, p_amp, w, prop = self.sp_amplitudes_from_fields(Efield, Hfield, i, j, ky_index)
                            if mode == "zeroth order":
                                order_indices = [oi]
                            else:
                                order_indices = np.where((w > 0) & prop)[0]
                            for m in order_indices:
                                out = np.array([s_amp[m], p_amp[m]], dtype=complex)
                                amp_R = np.vdot(Rv, out)
                                amp_L = np.vdot(Lv, out)
                                Rout[i, j] += w[m] * abs(amp_R)**2
                                Lout[i, j] += w[m] * abs(amp_L)**2
                    return Rout, Lout

                RR_out, LR_out = decompose_input("RCP")
                RL_out, LL_out = decompose_input("LCP")
                DoCP_RCP_in = safe_contrast(RR_out, LR_out)
                DoCP_LCP_in = safe_contrast(RL_out, LL_out)
            except Exception:
                pass

        out = {
            "R_RCP": R_RCP, "R_LCP": R_LCP,
            "T_RCP": T_RCP, "T_LCP": T_LCP,
            "A_RCP": A_RCP, "A_LCP": A_LCP,
            "CD": CD,
            "RR_out": RR_out, "LR_out": LR_out,
            "RL_out": RL_out, "LL_out": LL_out,
            "DoCP_RCP_in": DoCP_RCP_in,
            "DoCP_LCP_in": DoCP_LCP_in,
        }
        self._cache[key] = out
        return out

    def cp_maps_for_ky(self, ky_index):
        mode = self.order_mode_box.currentText()
        key = ("cp_b0", ky_index, mode)
        if key in self._cache:
            return self._cache[key]

        Ep, Hp = self.reflected_fields_for_pol("p", ky_index)
        Es, Hs = self.reflected_fields_for_pol("s", ky_index)

        energies = self.data["energies"]
        kxs = self.data["kxs"]
        shape = (len(energies), len(kxs))

        RR = np.zeros(shape, dtype=float)
        LL = np.zeros(shape, dtype=float)
        RL = np.zeros(shape, dtype=float)
        LR = np.zeros(shape, dtype=float)

        # Basis vector order is [s, p]. Convention: R = (p + i*s)/sqrt(2), L = (p - i*s)/sqrt(2).
        Rv = np.array([1j, 1.0], dtype=complex) / np.sqrt(2.0)
        Lv = np.array([-1j, 1.0], dtype=complex) / np.sqrt(2.0)

        oi = self.zero_order_index()

        for i in range(len(energies)):
            for j in range(len(kxs)):
                # For s incidence: output amplitudes [s_out, p_out].
                ss, ps_from_s, ws, prop = self.sp_amplitudes_from_fields(Es, Hs, i, j, ky_index, side="reflection")
                # For p incidence: output amplitudes [s_out, p_out].
                sp_from_p, pp, wp, prop2 = self.sp_amplitudes_from_fields(Ep, Hp, i, j, ky_index, side="reflection")

                w = 0.5*(ws + wp)
                if mode == "zeroth order":
                    order_indices = [oi]
                else:
                    order_indices = np.where((w > 0) & prop & prop2)[0]

                for m in order_indices:
                    # Rows: output [s,p]. Columns: input [s,p].
                    J = np.array([
                        [ss[m],      sp_from_p[m]],
                        [ps_from_s[m], pp[m]],
                    ], dtype=complex)

                    out_R = J @ Rv
                    out_L = J @ Lv

                    # Analyze output in circular basis too.
                    amp_RR = np.vdot(Rv, out_R)
                    amp_LR = np.vdot(Lv, out_R)
                    amp_RL = np.vdot(Rv, out_L)
                    amp_LL = np.vdot(Lv, out_L)

                    RR[i, j] += w[m] * abs(amp_RR)**2
                    LR[i, j] += w[m] * abs(amp_LR)**2
                    RL[i, j] += w[m] * abs(amp_RL)**2
                    LL[i, j] += w[m] * abs(amp_LL)**2

        # Plain reflected power for synthetic RCP/LCP input is the sum over output circular channels.
        R_RCP = RR + LR
        R_LCP = RL + LL
        CD = safe_contrast(R_RCP, R_LCP)
        DoCP_RCP = safe_contrast(RR, LR)
        DoCP_LCP = safe_contrast(RL, LL)
        out = {
            "RR": RR, "LL": LL, "RL": RL, "LR": LR,
            "R_RCP": R_RCP, "R_LCP": R_LCP,
            "CD": CD, "DoCP_RCP": DoCP_RCP, "DoCP_LCP": DoCP_LCP,
        }
        self._cache[key] = out
        return out

    def _parse_float_lineedit(self, widget, default):
        try:
            return float(str(widget.text()).strip())
        except Exception:
            return float(default)

    def custom_jones_vector(self, role="incident"):
        """Return normalized custom Jones vector in [s,p] basis.

        Convention: [s,p] = [r exp(i phi), 1], normalized.
        role is "incident" or "analyzer" and selects the corresponding GUI fields.
        """
        role = str(role).lower()
        if role.startswith("ana") or role.startswith("det") or role.startswith("out"):
            r = self._parse_float_lineedit(self.ana_custom_ratio_edit, 1.0)
            phi_deg = self._parse_float_lineedit(self.ana_custom_phase_edit, 90.0)
        else:
            r = self._parse_float_lineedit(self.inc_custom_ratio_edit, 1.0)
            phi_deg = self._parse_float_lineedit(self.inc_custom_phase_edit, 90.0)
        r = max(float(r), 0.0)
        phi = np.deg2rad(float(phi_deg))
        v = np.array([r*np.exp(1j*phi), 1.0+0j], dtype=complex)
        n = np.sqrt(np.vdot(v, v).real)
        if not np.isfinite(n) or n <= 0:
            return np.array([0.0, 1.0], dtype=complex)
        return v / n

    def polarization_cache_label(self, label, role="incident"):
        """Stable cache label; includes custom Jones parameters so changing them recomputes."""
        lab = str(label).strip()
        if lab.lower().startswith("custom"):
            if str(role).lower().startswith("ana"):
                r = self._parse_float_lineedit(self.ana_custom_ratio_edit, 1.0)
                ph = self._parse_float_lineedit(self.ana_custom_phase_edit, 90.0)
                return f"CustomAnalyzer(r={r:.8g},phi={ph:.8g})"
            r = self._parse_float_lineedit(self.inc_custom_ratio_edit, 1.0)
            ph = self._parse_float_lineedit(self.inc_custom_phase_edit, 90.0)
            return f"CustomIncident(r={r:.8g},phi={ph:.8g})"
        return lab

    def basis_vector(self, label, role="incident"):
        """Return vector in [s, p] basis using convention RCP=(p+i*s)/sqrt(2)."""
        lab = str(label).lower().strip()
        if lab.startswith("custom") or lab in ["jones", "custom_jones"]:
            return self.custom_jones_vector(role=role)
        if lab.startswith("s"):
            return np.array([1.0, 0.0], dtype=complex)
        if lab.startswith("p"):
            return np.array([0.0, 1.0], dtype=complex)
        if lab.startswith("rcp"):
            return np.array([1j, 1.0], dtype=complex) / np.sqrt(2.0)
        if lab.startswith("lcp"):
            return np.array([-1j, 1.0], dtype=complex) / np.sqrt(2.0)
        raise ValueError(f"Unknown polarization basis: {label}")

    def incident_a0_vector(self, incident, E_eV, kx_inc, ky_inc, nG):
        """Build grcwa incident a0 vector for the selected incident polarization.

        The incident vector is in the same internal two-block basis as b0:
        first nG entries and second nG entries, with the incident order set at G=(0,0).
        This follows grcwa MakeExcitationPlanewave for forward incidence.
        """
        vin = self.basis_vector(incident, role="incident")  # [s, p]
        s_amp = complex(vin[0])
        p_amp = complex(vin[1])
        a0 = np.zeros(2*nG, dtype=complex)
        m0 = self.zero_order_index()

        k0 = 2*np.pi*float(E_eV)/HC_EV_UM
        n_top = self.n_air()
        nk0 = n_top * k0
        kt = np.hypot(float(kx_inc), float(ky_inc))
        if nk0 <= 0:
            theta = 0.0
        else:
            theta = np.arcsin(np.clip(kt/nk0, -1.0, 1.0))
        phi_ang = np.arctan2(float(ky_inc), float(kx_inc)) if kt > 1e-15 else 0.0

        a0[m0] = (-s_amp*np.cos(theta)*np.cos(phi_ang) - p_amp*np.sin(phi_ang))
        a0[m0+nG] = (-s_amp*np.cos(theta)*np.sin(phi_ang) + p_amp*np.cos(phi_ang))
        return a0

    def reflected_b0_for_incident(self, incident, ky_index, iE, ikx, b0_key="b0"):
        """Coherently synthesize b0 for incident s/p/RCP/LCP from saved p and s b0."""
        if self.pol_index("p") is None or self.pol_index("s") is None:
            raise ValueError("This analysis requires saved p and s runs. Re-run solver with Pol. basis = p and s.")
        if b0_key not in self.data.files:
            raise ValueError(f"Missing {b0_key} in NPZ.")
        vin = self.basis_vector(incident, role="incident")  # [s, p]
        is_idx = self.pol_index("s")
        ip_idx = self.pol_index("p")
        b_s = np.asarray(self.data[b0_key][is_idx, iE, ikx, ky_index, :], dtype=complex)
        b_p = np.asarray(self.data[b0_key][ip_idx, iE, ikx, ky_index, :], dtype=complex)
        return vin[0]*b_s + vin[1]*b_p

    def reflected_power_poynting_for(self, incident, ky_index, b0_key="b0"):
        """Total reflected power from b0 using the same Poynting-flux formula as RT_Solve.

        This is for Analyzer=None. It is the correct replacement for the older
        |E_s|^2+|E_p|^2 estimate when comparing against RT_Solve saved R.
        """
        mode = self.order_mode_box.currentText()
        key = ("refl_poynting", self.polarization_cache_label(incident, "incident"), ky_index, b0_key, mode)
        if key in self._cache:
            return self._cache[key]
        cached = self._npz_cached_map_for_cache_key(key)
        if cached is not None:
            return cached

        energies = np.asarray(self.data["energies"], dtype=float)
        kxs_inc = np.asarray(self.data["kxs"], dtype=float)
        ky_inc = float(self.data["kys"][ky_index])
        nG = len(np.asarray(self.data["G_orders"])) if "G_orders" in self.data.files else np.asarray(self.data[b0_key]).shape[-1]//2
        m0 = self.zero_order_index()
        eps_top = self.n_air()**2
        out = np.full((len(energies), len(kxs_inc)), np.nan, dtype=float)

        for i, Eev in enumerate(energies):
            omega = 2*np.pi*float(Eev)/HC_EV_UM
            for j, kx0 in enumerate(kxs_inc):
                b = self.reflected_b0_for_incident(incident, ky_index, i, j, b0_key=b0_key)
                if b.size < 2*nG:
                    continue
                kx, ky, kz_ref, kz_abs, prop, nk0 = self.order_geometry(Eev, kx0, ky_inc)
                q0 = kz_abs.astype(complex)
                # branch cut as in grcwa: duplicate q for the two transverse blocks.
                q = np.concatenate((q0, q0))
                # Avoid divide-by-zero for evanescent/cutoff orders; they are nonpropagating anyway.
                q[np.abs(q) < 1e-15] = 1e-15 + 0j
                phi = np.eye(2*nG, dtype=complex)
                kp = make_kp_matrix_uniform(omega, eps_top, kx, ky)
                a0 = self.incident_a0_vector(incident, Eev, kx0, ky_inc, nG)

                _, backward = get_z_poynting_flux_np(a0, b, omega, kp, phi, q, byorder=1)
                kt = np.hypot(float(kx0), float(ky_inc))
                cos_th = np.sqrt(max(1.0 - (kt/max(nk0, 1e-30))**2, 0.0))
                normalization = self.n_air() / max(cos_th, 1e-30)
                R_orders = np.real(-backward) * normalization

                if mode == "zeroth order":
                    val = R_orders[m0]
                else:
                    # Sum propagating reflected orders only.  Evanescent orders carry no far-field power.
                    val = np.nansum(np.where(prop, R_orders, 0.0))
                out[i, j] = float(val)

        self._cache[key] = out
        return out

    def reflected_power_for(self, incident, analyzer, ky_index, b0_key="b0"):
        """Reflected power for an incident polarization and optional output analyzer.

        Analyzer=None now uses the true RCWA Poynting-flux formula from b0,
        matching RT_Solve. Polarization analyzers still use the reconstructed
        complex far-field Jones amplitudes.
        """
        analyzer = str(analyzer)
        if analyzer.lower().startswith("no"):
            return self.reflected_power_poynting_for(incident, ky_index, b0_key=b0_key)

        mode = self.order_mode_box.currentText()
        key = ("refl_power", self.polarization_cache_label(incident, "incident"), self.polarization_cache_label(analyzer, "analyzer"), ky_index, b0_key, mode)
        if key in self._cache:
            return self._cache[key]
        cached = self._npz_cached_map_for_cache_key(key)
        if cached is not None:
            return cached

        if self.pol_index("p") is None or self.pol_index("s") is None:
            raise ValueError("This analysis requires saved p and s runs. Re-run solver with Pol. basis = p and s.")
        if b0_key not in self.data.files:
            raise ValueError(f"Missing {b0_key} in NPZ.")

        Ep, Hp = self.reflected_fields_for_pol("p", ky_index, b0_key=b0_key)
        Es, Hs = self.reflected_fields_for_pol("s", ky_index, b0_key=b0_key)

        energies = np.asarray(self.data["energies"], dtype=float)
        kxs = np.asarray(self.data["kxs"], dtype=float)
        out_map = np.zeros((len(energies), len(kxs)), dtype=float)
        oi = self.zero_order_index()

        vin = self.basis_vector(incident, role="incident")
        vout = self.basis_vector(analyzer, role="analyzer")

        for i in range(len(energies)):
            for j in range(len(kxs)):
                # For s incidence: output amplitudes [s_out, p_out]
                ss, ps_from_s, ws, prop = self.sp_amplitudes_from_fields(Es, Hs, i, j, ky_index, side="reflection")
                # For p incidence: output amplitudes [s_out, p_out]
                sp_from_p, pp, wp, prop2 = self.sp_amplitudes_from_fields(Ep, Hp, i, j, ky_index, side="reflection")
                w = 0.5 * (ws + wp)
                if mode == "zeroth order":
                    order_indices = [oi]
                else:
                    order_indices = np.where((w > 0) & prop & prop2)[0]
                val = 0.0
                for m in order_indices:
                    J = np.array([[ss[m], sp_from_p[m]], [ps_from_s[m], pp[m]]], dtype=complex)
                    eout = J @ vin
                    amp = np.vdot(vout, eout)
                    val += w[m] * abs(amp)**2
                out_map[i, j] = val

        self._cache[key] = out_map
        return out_map

    def transmitted_aN_for_incident(self, incident, ky_index, iE, ikx, aN_key="aN"):
        """Coherently synthesize aN for incident s/p/RCP/LCP from saved p and s aN."""
        if self.pol_index("p") is None or self.pol_index("s") is None:
            raise ValueError("This analysis requires saved p and s runs. Re-run solver with Pol. basis = p and s.")
        if aN_key not in self.data.files:
            raise ValueError(f"Missing {aN_key} in NPZ. Re-run solver v4.3+ to save transmitted complex amplitudes aN.")
        vin = self.basis_vector(incident, role="incident")  # [s, p]
        is_idx = self.pol_index("s")
        ip_idx = self.pol_index("p")
        a_s = np.asarray(self.data[aN_key][is_idx, iE, ikx, ky_index, :], dtype=complex)
        a_p = np.asarray(self.data[aN_key][ip_idx, iE, ikx, ky_index, :], dtype=complex)
        return vin[0]*a_s + vin[1]*a_p

    def transmitted_power_poynting_for(self, incident, ky_index, aN_key="aN"):
        """Total transmitted power from aN using the same Poynting-flux formula as RT_Solve.

        This is the correct Analyzer=None transmission power. It uses the bottom
        homogeneous medium, aN as the forward outgoing amplitude, and bN=0.
        """
        mode = self.order_mode_box.currentText()
        key = ("trans_poynting", self.polarization_cache_label(incident, "incident"), ky_index, aN_key, mode)
        if key in self._cache:
            return self._cache[key]
        cached = self._npz_cached_map_for_cache_key(key)
        if cached is not None:
            return cached

        if aN_key not in self.data.files:
            raise ValueError(f"Missing {aN_key} in NPZ. Re-run solver v4.3+ to save transmitted complex amplitudes aN.")

        energies = np.asarray(self.data["energies"], dtype=float)
        kxs_inc = np.asarray(self.data["kxs"], dtype=float)
        ky_inc = float(self.data["kys"][ky_index])
        nG = len(np.asarray(self.data["G_orders"])) if "G_orders" in self.data.files else np.asarray(self.data[aN_key]).shape[-1]//2
        m0 = self.zero_order_index()
        eps_bot = self.n_glass()**2
        out = np.full((len(energies), len(kxs_inc)), np.nan, dtype=float)

        for i, Eev in enumerate(energies):
            omega = 2*np.pi*float(Eev)/HC_EV_UM
            for j, kx0 in enumerate(kxs_inc):
                a = self.transmitted_aN_for_incident(incident, ky_index, i, j, aN_key=aN_key)
                if a.size < 2*nG:
                    continue
                kx, ky, kz_tr, kz_abs, prop, nk0_bot = self.side_order_geometry(Eev, kx0, ky_inc, side="transmission")
                q0 = kz_abs.astype(complex)
                q = np.concatenate((q0, q0))
                q[np.abs(q) < 1e-15] = 1e-15 + 0j
                phi = np.eye(2*nG, dtype=complex)
                kp = make_kp_matrix_uniform(omega, eps_bot, kx, ky)
                bN = np.zeros_like(a)

                forward, _ = get_z_poynting_flux_np(a, bN, omega, kp, phi, q, byorder=1)

                # Normalize by incident power in the top medium, same factor as RT_Solve(normalize=1).
                k0 = 2*np.pi*float(Eev)/HC_EV_UM
                nk0_top = self.n_air() * k0
                kt_inc = np.hypot(float(kx0), float(ky_inc))
                cos_th = np.sqrt(max(1.0 - (kt_inc/max(nk0_top, 1e-30))**2, 0.0))
                normalization = self.n_air() / max(cos_th, 1e-30)
                T_orders = np.real(forward) * normalization

                if mode == "zeroth order":
                    val = T_orders[m0]
                else:
                    val = np.nansum(np.where(prop, T_orders, 0.0))
                out[i, j] = float(val)

        self._cache[key] = out
        return out

    def transmitted_power_for(self, incident, analyzer, ky_index, aN_key="aN"):
        """Transmitted power for selected incident and optional output analyzer using saved aN.

        Analyzer=None uses true RCWA Poynting flux from aN, matching RT_Solve.
        Polarization analyzers use reconstructed complex far-field amplitudes.
        """
        analyzer = str(analyzer)
        if analyzer.lower().startswith("no"):
            return self.transmitted_power_poynting_for(incident, ky_index, aN_key=aN_key)

        mode = self.order_mode_box.currentText()
        key = ("trans_power", self.polarization_cache_label(incident, "incident"), self.polarization_cache_label(analyzer, "analyzer"), ky_index, aN_key, mode)
        if key in self._cache:
            return self._cache[key]
        cached = self._npz_cached_map_for_cache_key(key)
        if cached is not None:
            return cached

        if self.pol_index("p") is None or self.pol_index("s") is None:
            raise ValueError("This analysis requires saved p and s runs. Re-run solver with Pol. basis = p and s.")
        if aN_key not in self.data.files or self.data[aN_key] is None:
            raise ValueError(f"Missing {aN_key} in NPZ. Re-run solver v4.3+ to save transmitted complex amplitudes aN.")

        Ep, Hp = self.transmitted_fields_for_pol("p", ky_index, aN_key=aN_key)
        Es, Hs = self.transmitted_fields_for_pol("s", ky_index, aN_key=aN_key)

        energies = np.asarray(self.data["energies"], dtype=float)
        kxs = np.asarray(self.data["kxs"], dtype=float)
        out_map = np.zeros((len(energies), len(kxs)), dtype=float)
        oi = self.zero_order_index()

        vin = self.basis_vector(incident, role="incident")
        vout = self.basis_vector(analyzer, role="analyzer")

        for i in range(len(energies)):
            for j in range(len(kxs)):
                ss, ps_from_s, ws, prop = self.sp_amplitudes_from_fields(Es, Hs, i, j, ky_index, side="transmission")
                sp_from_p, pp, wp, prop2 = self.sp_amplitudes_from_fields(Ep, Hp, i, j, ky_index, side="transmission")
                w = 0.5 * (ws + wp)
                if mode == "zeroth order":
                    order_indices = [oi]
                else:
                    order_indices = np.where((w > 0) & prop & prop2)[0]
                val = 0.0
                for m in order_indices:
                    J = np.array([[ss[m], sp_from_p[m]], [ps_from_s[m], pp[m]]], dtype=complex)
                    eout = J @ vin
                    amp = np.vdot(vout, eout)
                    val += w[m] * abs(amp)**2
                out_map[i, j] = val

        self._cache[key] = out_map
        return out_map

    def current_region(self):
        if not hasattr(self, "region_box"):
            return "Sample"
        return self.region_box.currentText()

    def modal_key_for_region(self, base, region=None):
        r = (region or self.current_region()).strip().lower()
        if r.startswith("ref"):
            return base + "_ref"
        return base

    def saved_quantity_key_for_region(self, base, region=None):
        r = (region or self.current_region()).strip().lower()
        if r.startswith("ref"):
            return base + "ref"
        return base


    def _precomputed_order_suffix(self, order_mode=None):
        """Return solver precomputed order suffix ('zeroth' or 'all') for current order mode."""
        mode = self._normalize_order_mode(order_mode or self.order_mode_box.currentText())
        return "zeroth" if mode == "zeroth order" else "all"

    def _precomputed_region_suffix(self, region=None):
        reg = (region or self.current_region()).strip().lower()
        return "_ref" if reg.startswith("ref") else ""

    def _precomputed_common_map(self, quantity, incident=None, analyzer=None, ky_index=None, order_mode=None, region=None):
        """Return a precomputed solver map if the NPZ contains one.

        These are optional fast-path maps saved by solver v4.8+ for common
        circular incident total quantities. Returns None if unavailable or if
        the request is not one of the supported precomputed cases.
        """
        if self.data is None:
            return None
        incident = str(incident or self.pol_box.currentText()).strip()
        analyzer = str(analyzer if analyzer is not None else self.analyzer_box.currentText()).strip()
        q = self._normalize_quantity_name(quantity)
        ky = max(0, self.ky_box.currentIndex()) if ky_index is None else int(ky_index)
        order = self._precomputed_order_suffix(order_mode)
        reg_suf = self._precomputed_region_suffix(region)

        # v4.8 solver only precomputes RCP/LCP incident with no analyzer.
        if incident not in ["RCP", "LCP"]:
            return None
        if not analyzer.lower().startswith("no"):
            return None

        key = None
        if q in ["R", "R (b0 reconstructed)"]:
            key = f"R_{incident}_{order}{reg_suf}"
        elif q in ["T", "T (aN reconstructed)"]:
            key = f"T_{incident}_{order}{reg_suf}"
        elif q == "A":
            key = f"A_{incident}_{order}{reg_suf}"
        elif q == "DeltaR_over_R":
            # Delta quantities are Sample-vs-Reference by definition, so only
            # use the precomputed sample-normalized key irrespective of region.
            key = f"dRoverR_{incident}_{order}"
        elif q == "CD":
            key = f"R_CD_{order}{reg_suf}"
        else:
            return None

        if key not in self.data.files:
            return None
        arr = np.asarray(self.data[key], dtype=float)
        if arr.ndim == 3:
            if ky >= arr.shape[2]:
                return None
            arr = arr[:, :, ky]
        elif arr.ndim != 2:
            return None
        try:
            self.loading_label.setText(f"Using precomputed map: {key}")
        except Exception:
            pass
        return np.asarray(arr, dtype=float).copy()

    def selected_T_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        if str(analyzer).lower().startswith("no") and str(incident).lower().strip() in ["s", "p"]:
            return self.direct_saved_map_for_sp("T", incident, ky, region=self.current_region())
        return self.transmitted_power_for(incident, analyzer, ky, aN_key=self.modal_key_for_region("aN"))

    def selected_T_RT_saved_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        if not str(analyzer).lower().startswith("no"):
            raise ValueError("T (RT_Solve saved) is only defined for Analyzer=None.")
        if str(incident).lower().strip() not in ["s", "p"]:
            raise ValueError("T (RT_Solve saved) is only available for saved s or p incident runs. Use T (aN reconstructed) for synthetic RCP/LCP.")
        return self.direct_saved_map_for_sp("T", incident, ky, region=self.current_region())

    def selected_T_aN_reconstructed_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        return self.transmitted_power_for(incident, analyzer, ky, aN_key=self.modal_key_for_region("aN"))

    def selected_Tref_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        if str(analyzer).lower().startswith("no") and str(incident).lower().strip() in ["s", "p"]:
            return self.direct_saved_map_for_sp("T", incident, ky, region="Reference")
        return self.transmitted_power_for(incident, analyzer, ky, aN_key="aN_ref")

    def direct_saved_map_for_sp(self, quantity, incident, ky_index, region=None):
        """Direct saved R/T/A for literal s or p incident only, no analyzer. Region may be Sample or Reference."""
        ipol = self.pol_index(incident)
        if ipol is None:
            raise ValueError(f"Missing saved {incident} run.")
        key = self.saved_quantity_key_for_region(quantity, region)
        if key in self.data.files:
            return np.asarray(self.data[key][ipol, :, :, ky_index], dtype=float)
        if quantity == "A":
            Rkey = self.saved_quantity_key_for_region("R", region)
            Tkey = self.saved_quantity_key_for_region("T", region)
            return 1.0 - np.asarray(self.data[Rkey][ipol, :, :, ky_index], dtype=float) - np.asarray(self.data[Tkey][ipol, :, :, ky_index], dtype=float)
        raise ValueError(quantity)

    def selected_R_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        # Fast exact path for literal saved s/p total R: use solver RT_Solve output.
        # This avoids recomputing a Poynting map from b0 when no analyzer is requested.
        if str(analyzer).lower().startswith("no") and str(incident).lower().strip() in ["s", "p"]:
            return self.direct_saved_map_for_sp("R", incident, ky, region=self.current_region())
        return self.reflected_power_for(incident, analyzer, ky, b0_key=self.modal_key_for_region("b0"))

    def selected_R_RT_saved_map(self):
        """R exactly as saved by the solver RT_Solve path.

        This is available only for literal saved s or p incidence with Analyzer=None.
        It is kept as a diagnostic so it can be compared against the b0
        reconstruction map.
        """
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        if not str(analyzer).lower().startswith("no"):
            raise ValueError("R (RT_Solve saved) is only defined for Analyzer=None.")
        if str(incident).lower().strip() not in ["s", "p"]:
            raise ValueError("R (RT_Solve saved) is only available for saved s or p incident runs. Use R (b0 reconstructed) for synthetic RCP/LCP.")
        return self.direct_saved_map_for_sp("R", incident, ky, region=self.current_region())

    def selected_R_b0_reconstructed_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        return self.reflected_power_for(incident, analyzer, ky, b0_key=self.modal_key_for_region("b0"))

    def selected_Rref_map(self):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        if str(analyzer).lower().startswith("no") and str(incident).lower().strip() in ["s", "p"]:
            return self.direct_saved_map_for_sp("R", incident, ky, region="Reference")
        return self.reflected_power_for(incident, analyzer, ky, b0_key="b0_ref")

    def base_map(self, q):
        ky = max(0, self.ky_box.currentIndex())
        incident = self.pol_box.currentText()
        analyzer = self.analyzer_box.currentText()
        E = np.asarray(self.data["energies"], dtype=float)
        kx = np.asarray(self.data["kxs"], dtype=float)

        pre = self._precomputed_common_map(q, incident=incident, analyzer=analyzer, ky_index=ky, region=self.current_region())
        if pre is not None:
            return pre

        if q == "R":
            # Default remains b0 reconstruction so synthetic incident/analyzer states work.
            return self.selected_R_map()

        if q == "R (RT_Solve saved)":
            return self.selected_R_RT_saved_map()

        if q == "R (b0 reconstructed)":
            return self.selected_R_b0_reconstructed_map()

        if q == "1 - R":
            return 1.0 - self.selected_R_map()

        if q in ["DeltaR", "DeltaR_over_R"]:
            R = self.selected_R_map()
            Rref = self.selected_Rref_map()
            if q == "DeltaR":
                return R - Rref
            out = np.full_like(R, np.nan, dtype=float)
            m = np.isfinite(Rref) & (np.abs(Rref) > 1e-15)
            out[m] = (R[m] - Rref[m]) / Rref[m]
            return out

        if q == "d/dE of R":
            return derivative_map(self.selected_R_map(), E, kx, "d/dE")
        if q == "-d/dE of R":
            return derivative_map(self.selected_R_map(), E, kx, "-d/dE")
        if q == "|grad R|":
            return derivative_map(self.selected_R_map(), E, kx, "|grad")

        if q == "CD":
            # CD is defined from total reflected R for RCP/LCP input, independent of current analyzer.
            bkey = self.modal_key_for_region("b0")
            Rr = self.reflected_power_for("RCP", "None", ky, b0_key=bkey)
            Rl = self.reflected_power_for("LCP", "None", ky, b0_key=bkey)
            return safe_contrast(Rr, Rl)

        if q == "DoCP":
            # Degree of circular polarization of reflected output for selected incident.
            bkey = self.modal_key_for_region("b0")
            R_out = self.reflected_power_for(incident, "RCP", ky, b0_key=bkey)
            L_out = self.reflected_power_for(incident, "LCP", ky, b0_key=bkey)
            return safe_contrast(R_out, L_out)

        if q == "T":
            # Use transmitted aN with Poynting-flux normalization for Analyzer=None.
            return self.selected_T_map()

        if q == "T (RT_Solve saved)":
            return self.selected_T_RT_saved_map()

        if q == "T (aN reconstructed)":
            return self.selected_T_aN_reconstructed_map()

        if q == "A":
            # Absorption is a total quantity. For literal saved s/p, use the exact
            # solver-saved A when available. For synthetic circular states, use the
            # precomputed solver maps if present; otherwise reconstruct flux.
            if str(analyzer).lower().startswith("no") and str(incident).lower().strip() in ["s", "p"]:
                return self.direct_saved_map_for_sp("A", incident, ky, region=self.current_region())
            R = self.reflected_power_for(incident, "None", ky, b0_key=self.modal_key_for_region("b0"))
            aN_key = self.modal_key_for_region("aN")
            if aN_key in self.data.files:
                T = self.transmitted_power_for(incident, "None", ky, aN_key=aN_key)
            elif incident in ["s", "p"]:
                T = self.direct_saved_map_for_sp("T", incident, ky, region=self.current_region())
            else:
                raise ValueError("A for synthetic RCP/LCP requires saved transmitted complex amplitudes aN. Re-run solver v4.3+.")
            return 1.0 - R - T

        raise ValueError("Unsupported quantity.")

    def limits(self, Z, q):
        if not self.auto_scale_check.isChecked():
            try:
                return float(self.vmin_edit.text()), float(self.vmax_edit.text())
            except Exception:
                pass

        # Physically bounded polarization metrics: keep a stable absolute scale.
        if q in ["CD", "DoCP"]:
            return -1.0, 1.0

        # Phase-like quantities should always use the full wrapped range.
        if "phase" in str(q).lower():
            return -np.pi, np.pi

        # Difference/derivative maps: symmetric scale around zero.
        if q in ["DeltaR", "DeltaR_over_R", "d/dE of R", "-d/dE of R"]:
            return robust_limits(Z, symmetric=True, p_high=98)

        # Gradient maps are positive but usually have a few spikes.
        if q in ["|grad R|"]:
            return robust_limits(Z, positive=True, ignore_zero=True, p_low=2, p_high=98)

        # R/T/A are positive and often include zero masks at non-propagating orders.
        # Use high-contrast robust scaling for maps near unity: this clips low
        # edge/cutoff pixels and reveals the useful variation in the bright region.
        if q in ["R", "R (RT_Solve saved)", "R (b0 reconstructed)", "T", "A"]:
            return robust_limits(Z, positive=True, ignore_zero=True, p_low=5, p_high=99, high_contrast=True)

        # 1-R is already a contrast quantity, so ordinary robust positive scaling is better.
        if q in ["1 - R"]:
            return robust_limits(Z, positive=True, ignore_zero=True, p_low=2, p_high=98)

        return robust_limits(Z, p_low=2, p_high=98)


    def _normalize_order_mode(self, order_mode):
        if order_mode is None or str(order_mode).strip().lower() in ["", "current"]:
            return self.order_mode_box.currentText()
        t = str(order_mode).strip().lower().replace(" ", "_").replace("-", "_")
        if t in ["zeroth", "zero", "0", "zeroth_order"]:
            return "zeroth order"
        if t in ["all", "all_order", "all_orders", "all_propagating", "all_propagating_reflected_orders"]:
            return "all propagating reflected orders"
        raise ValueError(f"Unknown order mode: {order_mode}")

    def _normalize_quantity_name(self, quantity):
        q = str(quantity).strip()
        aliases = {
            "r": "R",
            "t": "T",
            "a": "A",
            "dr": "DeltaR",
            "deltar": "DeltaR",
            "dr/r": "DeltaR_over_R",
            "deltar_over_r": "DeltaR_over_R",
            "delta_r_over_r": "DeltaR_over_R",
            "1-r": "1 - R",
            "one_minus_r": "1 - R",
            "cd": "CD",
            "docp": "DoCP",
        }
        return aliases.get(q.lower(), q)

    def _set_combo_text_or_error(self, combo, text, what):
        text = str(text)
        idx = combo.findText(text, Qt.MatchFixedString)
        if idx < 0:
            opts = [combo.itemText(i) for i in range(combo.count())]
            raise ValueError(f"Unknown {what}: {text}. Options: {opts}")
        combo.setCurrentIndex(idx)

    def signal_map(self, incident, quantity, analyzer="None", order_mode=None, region=None):
        """Return a map for a chosen signal without changing the visible controls.

        Use in the Python box as S('s','R','RCP','zeroth','Sample') or S(s,R,RCP,zeroth,Reference).
        incident: s, p, RCP, LCP
        quantity: R, T, A, DeltaR, DeltaR_over_R, CD, DoCP, etc.
        analyzer: None, s, p, RCP, LCP
        order_mode: zeroth/current/all/all_order
        region: Sample/current/Reference
        """
        old_inc = self.pol_box.currentText()
        old_q = self.quantity_box.currentText()
        old_ana = self.analyzer_box.currentText()
        old_order = self.order_mode_box.currentText()
        old_region = self.region_box.currentText() if hasattr(self, "region_box") else "Sample"
        try:
            inc = str(incident)
            q = self._normalize_quantity_name(quantity)
            ana = "None" if analyzer is None else str(analyzer)
            order = self._normalize_order_mode(order_mode)
            reg = old_region if region is None or str(region).strip().lower() in ["", "current"] else str(region)
            self._set_combo_text_or_error(self.pol_box, inc, "incident polarization")
            self._set_combo_text_or_error(self.quantity_box, q, "quantity")
            self._set_combo_text_or_error(self.analyzer_box, ana, "analyzer")
            self._set_combo_text_or_error(self.order_mode_box, order, "order mode")
            if hasattr(self, "region_box"):
                self._set_combo_text_or_error(self.region_box, reg, "region")
            return np.asarray(self.base_map(q), dtype=float).copy()
        finally:
            self._set_combo_text_or_error(self.pol_box, old_inc, "incident polarization")
            self._set_combo_text_or_error(self.quantity_box, old_q, "quantity")
            self._set_combo_text_or_error(self.analyzer_box, old_ana, "analyzer")
            self._set_combo_text_or_error(self.order_mode_box, old_order, "order mode")
            if hasattr(self, "region_box"):
                self._set_combo_text_or_error(self.region_box, old_region, "region")

    def assign_current_signal(self):
        if self.data is None:
            QMessageBox.information(self, "No file", "Load an NPZ first.")
            return
        self.begin_busy("Assigning current signal...")
        try:
            q = self.quantity_box.currentText()
            Z = np.asarray(self.base_map(q), dtype=float).copy()
            self.custom_var_counter += 1
            name = f"v{self.custom_var_counter}"
            self.custom_vars[name] = Z
            spec = {
                "name": name,
                "incident": self.pol_box.currentText(),
                "quantity": q,
                "analyzer": self.analyzer_box.currentText(),
                "order_mode": self.order_mode_box.currentText(),
                "region": self.region_box.currentText(),
            }
            self.custom_var_specs.append(spec)
            label = self._custom_signal_label(spec)
            self.custom_signal_list.addItem(label)
            self.meta_text.setPlainText(f"Assigned {label}\nshape={Z.shape}")
            self.end_busy("Signal assigned")
        except Exception as e:
            self.end_busy("Assign failed")
            QMessageBox.critical(self, "Assign signal error", str(e))

    def clear_custom_signals(self):
        self.custom_vars.clear()
        self.custom_var_specs.clear()
        self.custom_var_counter = 0
        self.custom_signal_list.clear()

    def _custom_signal_label(self, spec):
        return (
            f"{spec.get('name', 'v?')} = S('{spec.get('incident', 's')}', "
            f"'{spec.get('quantity', 'R')}', '{spec.get('analyzer', 'None')}', "
            f"'{spec.get('order_mode', 'zeroth order')}', '{spec.get('region', 'Sample')}')"
        )

    def _preprocess_custom_expr(self, expr):
        """Allow shorthand:
        (s,R,RCP,zeroth) -> S(s,R,RCP,zeroth)
        (s,R,RCP,zeroth,Reference) -> S(s,R,RCP,zeroth,Reference)
        """
        pattern5 = re.compile(
            r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
        )
        expr = pattern5.sub(r"S(\1, \2, \3, \4, \5)", expr)
        pattern4 = re.compile(
            r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
        )
        return pattern4.sub(r"S(\1, \2, \3, \4)", expr)

    def evaluate_custom_expression(self):
        if self.data is None:
            raise ValueError("Load an NPZ first.")
        expr_raw = self.custom_expr_edit.toPlainText().strip()
        if not expr_raw:
            raise ValueError("Write a Python expression first.")
        expr = self._preprocess_custom_expr(expr_raw)
        env = {
            "np": np,
            "div": safe_divide,
            "safe_divide": safe_divide,
            "S": self.signal_map,
            "sig": self.signal_map,
            "nan": np.nan,
            "pi": np.pi,
            # tokens for shorthand S(s,R,RCP,zeroth)
            "s": "s", "p": "p", "RCP": "RCP", "LCP": "LCP", "None": "None",
            "Custom": "Custom Jones", "Jones": "Custom Jones", "CustomJones": "Custom Jones",
            "R": "R", "T": "T", "A": "A", "DeltaR": "DeltaR", "DeltaR_over_R": "DeltaR_over_R",
            "CD": "CD", "DoCP": "DoCP", "zeroth": "zeroth", "all": "all", "all_order": "all_order",
            "Sample": "Sample", "Reference": "Reference", "sample": "Sample", "reference": "Reference",
        }
        env.update(self.custom_vars)
        # Give NumPy/matplotlib enough harmless builtins to operate.
        # The previous empty builtins dictionary could trigger a KeyError
        # named "__import__" during some NumPy operations even for simple
        # expressions like (v1-v5)/(v2-v4).
        safe_builtins = {
            "__import__": __import__,
            "abs": abs, "min": min, "max": max, "sum": sum,
            "len": len, "float": float, "int": int, "complex": complex,
            "round": round, "range": range,
        }
        # Custom ratios such as (v1-v5)/(v2-v4) can legitimately create
        # divide-by-zero regions.  Do not let inf/overflow values crash the
        # plotting backend; keep them as NaN masks.  For especially delicate
        # ratios the expression box also exposes div(a, b, eps=...).
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            out = eval(expr, {"__builtins__": safe_builtins}, env)
        Z = sanitize_map(np.asarray(out, dtype=float))
        target_shape = (len(self.data["energies"]), len(self.data["kxs"]))
        if Z.shape == ():
            Z = np.full(target_shape, float(Z), dtype=float)
            Z = sanitize_map(Z)
        if Z.shape != target_shape:
            raise ValueError(f"Expression returned shape {Z.shape}, expected {target_shape}.")
        if not np.any(np.isfinite(Z)):
            raise ValueError("Expression produced no finite values. If this is a ratio, try div(numerator, denominator, eps=1e-9).")
        return Z, expr_raw

    def plot_custom_expression(self):
        """Evaluate the Python/custom signal expression and plot it using the
        same Plot mode / Cut E / Cut kx controls as normal quantities.
        """
        if self.data is None:
            return
        self.begin_busy("Computing custom Python expression...")
        try:
            self.meta_text.setPlainText("Computing custom expression, please wait...")
            QApplication.processEvents()
            Z, expr_raw = self.evaluate_custom_expression()
            E = np.asarray(self.data["energies"], dtype=float)
            kx = np.asarray(self.data["kxs"], dtype=float)
            ky_index = max(0, self.ky_box.currentIndex())
            ky_val = float(self.data["kys"][ky_index])
            vmin, vmax = robust_limits(Z, p_low=2, p_high=98)
            iE, ikx = self.selected_cut_indices(E, kx)
            mode = self.plot_mode_box.currentText()
            custom_label = "custom"
            custom_title = f"Custom expression: {expr_raw}"

            self.fig.clear()

            if mode == "2D map":
                ax = self.fig.add_subplot(111)
                extent = [kx[0], kx[-1], E[0], E[-1]]
                im = ax.imshow(np.ma.masked_invalid(Z), origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
                if self.show_dashed_lines_check.isChecked():
                    ax.axhline(float(E[iE]), linestyle="--", linewidth=1.0)
                    ax.axvline(float(kx[ikx]), linestyle="--", linewidth=1.0)
                ax.set_title(f"{custom_title}\nky={ky_val:.4g} um$^{{-1}}$")
                ax.set_xlabel("kx (um$^{-1}$)")
                ax.set_ylabel("Energy (eV)")
                self.fig.colorbar(im, ax=ax, label=custom_label)

            elif mode == "E cut: value vs kx":
                ax = self.fig.add_subplot(111)
                y = np.asarray(Z[iE, :], dtype=float)
                ax.plot(kx, y, marker="o", markersize=3)
                ax.set_title(f"{custom_title}\nE cut: E={E[iE]:.5g} eV, ky={ky_val:.4g} um$^{{-1}}$")
                ax.set_xlabel("kx (um$^{-1}$)")
                ax.set_ylabel(custom_label)
                ax.grid(True, alpha=0.3)

            elif mode == "kx cut: value vs E":
                ax = self.fig.add_subplot(111)
                y = np.asarray(Z[:, ikx], dtype=float)
                ax.plot(E, y, marker="o", markersize=3)
                ax.set_title(f"{custom_title}\nkx cut: kx={kx[ikx]:.5g} um$^{{-1}}$, ky={ky_val:.4g} um$^{{-1}}$")
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel(custom_label)
                ax.grid(True, alpha=0.3)

            else:  # 2D map + both cuts
                gs = self.fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])
                ax_map = self.fig.add_subplot(gs[:, 0])
                ax_e = self.fig.add_subplot(gs[0, 1])
                ax_k = self.fig.add_subplot(gs[1, 1])

                extent = [kx[0], kx[-1], E[0], E[-1]]
                im = ax_map.imshow(np.ma.masked_invalid(Z), origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
                if self.show_dashed_lines_check.isChecked():
                    ax_map.axhline(float(E[iE]), linestyle="--", linewidth=1.0)
                    ax_map.axvline(float(kx[ikx]), linestyle="--", linewidth=1.0)
                ax_map.set_title(f"{custom_title}\nky={ky_val:.4g} um$^{{-1}}$")
                ax_map.set_xlabel("kx (um$^{-1}$)")
                ax_map.set_ylabel("Energy (eV)")
                self.fig.colorbar(im, ax=ax_map, label=custom_label, fraction=0.046, pad=0.04)

                ax_e.plot(kx, np.asarray(Z[iE, :], dtype=float), marker="o", markersize=2.5)
                ax_e.set_title(f"E cut: E={E[iE]:.5g} eV")
                ax_e.set_xlabel("kx (um$^{-1}$)")
                ax_e.set_ylabel(custom_label)
                ax_e.grid(True, alpha=0.3)

                ax_k.plot(E, np.asarray(Z[:, ikx], dtype=float), marker="o", markersize=2.5)
                ax_k.set_title(f"kx cut: kx={kx[ikx]:.5g} um$^{{-1}}$")
                ax_k.set_xlabel("Energy (eV)")
                ax_k.set_ylabel(custom_label)
                ax_k.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.current_map = Z
            self.current_quantity = "custom: " + expr_raw
            self.meta_text.setPlainText(
                json.dumps({
                    "viewer_version": VIEWER_VERSION,
                    "custom_expression": expr_raw,
                    "preprocessed_expression": self._preprocess_custom_expr(expr_raw),
                    "plot_mode": mode,
                    "cut_E_eV": float(E[iE]),
                    "cut_kx_um^-1": float(kx[ikx]),
                    "available_variables": sorted(self.custom_vars.keys()),
                    "helper": "S(incident, quantity, analyzer='None', order='current')",
                    "shape": list(Z.shape),
                }, indent=2)
            )
            self.end_busy("Custom expression plotted")
        except Exception as e:
            self.end_busy("Custom expression failed")
            QMessageBox.critical(self, "Custom expression error", str(e))
            try:
                self.update_metadata()
            except Exception:
                pass

    def update_metadata(self):
        if self.data is None:
            return
        info = {
            "viewer_version": VIEWER_VERSION,
            "loaded_file": self.path,
            "pol_labels": [str(x) for x in self.pol_labels()],
            "has_p_and_s": self.pol_index("p") is not None and self.pol_index("s") is not None,
            "has_b0": "b0" in self.data.files,
            "has_aN": "aN" in self.data.files,
            "T_None_uses_aN_Poynting_flux": True,
            "has_direct_RCP_LCP": self.has_direct_cp(),
            "can_synthesize_circular_from_p_s_b0": (self.pol_index("p") is not None and self.pol_index("s") is not None and "b0" in self.data.files),
            "can_synthesize_transmission_from_p_s_aN": (self.pol_index("p") is not None and self.pol_index("s") is not None and "aN" in self.data.files),
            "has_E_top": "E_top" in self.data.files,
            "has_H_top": "H_top" in self.data.files,
            "G_orders_shape": list(self.data["G_orders"].shape) if "G_orders" in self.data.files else None,
            "zero_order_index": self.zero_order_index() if self.data is not None else None,
            "incident": self.pol_box.currentText() if hasattr(self, "pol_box") else None,
            "quantity": self.quantity_box.currentText() if hasattr(self, "quantity_box") else None,
            "analyzer": self.analyzer_box.currentText() if hasattr(self, "analyzer_box") else None,
            "custom_jones": {
                "incident_vector_sp": [str(x) for x in self.custom_jones_vector("incident")] if hasattr(self, "inc_custom_ratio_edit") else None,
                "analyzer_vector_sp": [str(x) for x in self.custom_jones_vector("analyzer")] if hasattr(self, "ana_custom_ratio_edit") else None,
                "incident_ratio_Es_over_Ep": self.inc_custom_ratio_edit.text() if hasattr(self, "inc_custom_ratio_edit") else None,
                "incident_phase_deg": self.inc_custom_phase_edit.text() if hasattr(self, "inc_custom_phase_edit") else None,
                "analyzer_ratio_Es_over_Ep": self.ana_custom_ratio_edit.text() if hasattr(self, "ana_custom_ratio_edit") else None,
                "analyzer_phase_deg": self.ana_custom_phase_edit.text() if hasattr(self, "ana_custom_phase_edit") else None,
            },
            "order_mode": self.order_mode_box.currentText(),
            "show_dashed_guide_lines": self.show_dashed_lines_check.isChecked() if hasattr(self, "show_dashed_lines_check") else True,
            "metadata": self.meta,
        }
        self.meta_text.setPlainText(json.dumps(info, indent=2))

    def nearest_index(self, arr, value, default_index=0):
        arr = np.asarray(arr, dtype=float)
        if value is None or not np.isfinite(value):
            return int(np.clip(default_index, 0, len(arr) - 1))
        return int(np.nanargmin(np.abs(arr - value)))

    def parse_optional_float(self, text):
        try:
            t = str(text).strip()
            if not t:
                return None
            return float(t)
        except Exception:
            return None

    def selected_cut_indices(self, E, kx):
        e_val = self.parse_optional_float(self.cut_E_edit.text())
        kx_val = self.parse_optional_float(self.cut_kx_edit.text())
        iE = self.nearest_index(E, e_val, default_index=len(E)//2)
        ikx = self.nearest_index(kx, kx_val, default_index=int(np.nanargmin(np.abs(kx))))
        return iE, ikx

    def draw_map_axis(self, ax, Z, E, kx, q, ky_val, vmin, vmax, iE=None, ikx=None, show_guides=True):
        extent = [kx[0], kx[-1], E[0], E[-1]]
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
        if show_guides and iE is not None and ikx is not None:
            ax.axhline(float(E[iE]), linestyle="--", linewidth=1.0)
            ax.axvline(float(kx[ikx]), linestyle="--", linewidth=1.0)
        inc = self.pol_box.currentText()
        ana = self.analyzer_box.currentText()
        if q in ["R", "1 - R", "DeltaR", "DeltaR_over_R", "d/dE of R", "-d/dE of R", "|grad R|"]:
            meas = f"Reflection: {inc} → {ana}" if ana != "None" else f"Reflection: {inc} → total"
        elif q == "T":
            meas = f"Transmission: {inc} → {ana}" if ana != "None" else f"Transmission: {inc} → total"
        elif q == "A":
            meas = f"Absorption: {inc}"
        else:
            meas = f"{q}: incident {inc}"
        title = f"{q} — {meas}, ky={ky_val:.4g} um$^{{-1}}$ [{self.order_mode_box.currentText()}]"
        ax.set_title(title)
        ax.set_xlabel("kx (um$^{-1}$)")
        ax.set_ylabel("Energy (eV)")
        return im

    def plot(self):
        if self.data is None or self._plotting:
            return
        self._plotting = True
        q = self.quantity_box.currentText()
        self.begin_busy(f"Computing {q} map...")
        try:
            self.meta_text.setPlainText("Computing plot, please wait...")
            QApplication.processEvents()
            Z = self.base_map(q)
            E = np.asarray(self.data["energies"], dtype=float)
            kx = np.asarray(self.data["kxs"], dtype=float)
            ky_index = max(0, self.ky_box.currentIndex())
            ky_val = float(self.data["kys"][ky_index])
            vmin, vmax = self.limits(Z, q)
            iE, ikx = self.selected_cut_indices(E, kx)
            mode = self.plot_mode_box.currentText()

            self.fig.clear()

            if mode == "2D map":
                ax = self.fig.add_subplot(111)
                im = self.draw_map_axis(
                    ax, Z, E, kx, q, ky_val, vmin, vmax,
                    iE=iE, ikx=ikx, show_guides=self.show_dashed_lines_check.isChecked()
                )
                self.fig.colorbar(im, ax=ax, label=q)

            elif mode == "E cut: value vs kx":
                ax = self.fig.add_subplot(111)
                y = np.asarray(Z[iE, :], dtype=float)
                ax.plot(kx, y, marker="o", markersize=3)
                ax.set_title(f"{q} vs kx at E={E[iE]:.5g} eV, ky={ky_val:.4g} um$^{{-1}}$")
                ax.set_xlabel("kx (um$^{-1}$)")
                ax.set_ylabel(q)
                ax.grid(True, alpha=0.3)

            elif mode == "kx cut: value vs E":
                ax = self.fig.add_subplot(111)
                y = np.asarray(Z[:, ikx], dtype=float)
                ax.plot(E, y, marker="o", markersize=3)
                ax.set_title(f"{q} vs Energy at kx={kx[ikx]:.5g} um$^{{-1}}$, ky={ky_val:.4g} um$^{{-1}}$")
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel(q)
                ax.grid(True, alpha=0.3)

            else:  # 2D map + both cuts
                gs = self.fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])
                ax_map = self.fig.add_subplot(gs[:, 0])
                ax_e = self.fig.add_subplot(gs[0, 1])
                ax_k = self.fig.add_subplot(gs[1, 1])
                im = self.draw_map_axis(
                    ax_map, Z, E, kx, q, ky_val, vmin, vmax,
                    iE=iE, ikx=ikx, show_guides=self.show_dashed_lines_check.isChecked()
                )
                self.fig.colorbar(im, ax=ax_map, label=q, fraction=0.046, pad=0.04)

                ax_e.plot(kx, np.asarray(Z[iE, :], dtype=float), marker="o", markersize=2.5)
                ax_e.set_title(f"E cut: E={E[iE]:.5g} eV")
                ax_e.set_xlabel("kx (um$^{-1}$)")
                ax_e.set_ylabel(q)
                ax_e.grid(True, alpha=0.3)

                ax_k.plot(E, np.asarray(Z[:, ikx], dtype=float), marker="o", markersize=2.5)
                ax_k.set_title(f"kx cut: kx={kx[ikx]:.5g} um$^{{-1}}$")
                ax_k.set_xlabel("Energy (eV)")
                ax_k.set_ylabel(q)
                ax_k.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.current_map = Z
            self.current_quantity = q
            self.update_metadata()
            self.end_busy("Plot ready")
        except Exception as e:
            self.end_busy("Plot failed")
            QMessageBox.critical(self, "Plot error", str(e))
            self.update_metadata()
        finally:
            self._plotting = False

    def _npz_json_string(self, key):
        """Return raw JSON string stored in the loaded NPZ, or None."""
        if self.data is None or key not in self.data.files:
            return None
        arr = self.data[key]
        try:
            value = arr.tolist()
        except Exception:
            value = arr
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _combo_text(self, combo):
        try:
            return combo.currentText()
        except Exception:
            return ""

    def _set_combo_if_present(self, combo, text):
        if text is None:
            return
        idx = combo.findText(str(text), Qt.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def viewer_config_dict(self):
        """Return a JSON-serializable snapshot of the current viewer state."""
        return {
            "viewer_config_version": VIEWER_VERSION,
            "npz_path": self.path,
            "controls": {
                "incident_polarization": self._combo_text(self.pol_box),
                "quantity": self._combo_text(self.quantity_box),
                "analyzer": self._combo_text(self.analyzer_box),
                "region": self._combo_text(self.region_box) if hasattr(self, "region_box") else "Sample",
                "ky_index": int(self.ky_box.currentIndex()) if hasattr(self, "ky_box") else 0,
                "ky_text": self._combo_text(self.ky_box),
                "order_mode": self._combo_text(self.order_mode_box),
                "plot_mode": self._combo_text(self.plot_mode_box),
                "cut_E_eV": self.cut_E_edit.text(),
                "cut_kx_um^-1": self.cut_kx_edit.text(),
                "auto_robust_color_scale": bool(self.auto_scale_check.isChecked()),
                "show_dashed_guide_lines": bool(self.show_dashed_lines_check.isChecked()),
                "vmin": self.vmin_edit.text(),
                "vmax": self.vmax_edit.text(),
                "inc_custom_ratio": self.inc_custom_ratio_edit.text() if hasattr(self, "inc_custom_ratio_edit") else "1.0",
                "inc_custom_phase_deg": self.inc_custom_phase_edit.text() if hasattr(self, "inc_custom_phase_edit") else "90.0",
                "analyzer_custom_ratio": self.ana_custom_ratio_edit.text() if hasattr(self, "ana_custom_ratio_edit") else "1.0",
                "analyzer_custom_phase_deg": self.ana_custom_phase_edit.text() if hasattr(self, "ana_custom_phase_edit") else "90.0",
            },
            "custom": {
                "expression": self.custom_expr_edit.toPlainText(),
                "signals": list(self.custom_var_specs),
            },
            "current_plot": {
                "quantity_label": self.current_quantity,
                "is_custom_expression": bool(isinstance(self.current_quantity, str) and self.current_quantity.startswith("custom:")),
            },
        }



    def _npz_cached_map_for_cache_key(self, key):
        """Load a viewer-computed 2D map that was previously written back into the NPZ.

        The write-back code stores maps using _cache_key_to_npz_name(key). This helper
        reverses that workflow at runtime: before recomputing an expensive signal,
        look for its saved viewer_cache__... array inside the loaded NPZ.
        """
        try:
            if self.data is None:
                return None
            name = self._cache_key_to_npz_name(key)
            if name in self.data.files:
                arr = np.asarray(self.data[name])
                target = (len(self.data["energies"]), len(self.data["kxs"]))
                if arr.ndim == 2 and arr.shape == target:
                    try:
                        self.loading_label.setText(f"Using saved viewer cache: {name}")
                    except Exception:
                        pass
                    out = np.asarray(arr, dtype=float).copy()
                    self._cache[key] = out
                    return out

            # Complex custom maps may be saved as __real and __imag. Recombine if present.
            nr = name + "__real"
            ni = name + "__imag"
            if nr in self.data.files and ni in self.data.files:
                ar = np.asarray(self.data[nr])
                ai = np.asarray(self.data[ni])
                target = (len(self.data["energies"]), len(self.data["kxs"]))
                if ar.ndim == 2 and ai.ndim == 2 and ar.shape == target and ai.shape == target:
                    out = ar + 1j*ai
                    self._cache[key] = out
                    return out
        except Exception:
            return None
        return None

    def _safe_npz_key_component(self, value):
        s = str(value)
        s = s.replace("μ", "u").replace("→", "to").replace("/", "_over_").replace("-", "minus")
        s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:80] if s else "x"

    def _cache_key_to_npz_name(self, key):
        """Create a deterministic NPZ key for a viewer-computed 2D map.

        Only 2D maps are written. Heavy intermediate objects such as E/H fields,
        dict caches, or modal arrays are deliberately skipped.
        """
        if isinstance(key, tuple):
            parts = [self._safe_npz_key_component(x) for x in key]
            return "viewer_cache__" + "__".join(parts)
        return "viewer_cache__" + self._safe_npz_key_component(key)

    def _collect_2d_viewer_maps_for_npz(self):
        """Return dict of safe NPZ key -> 2D float array for computed viewer maps."""
        maps = {}
        meta = []
        target_shape = None
        try:
            target_shape = (len(self.data["energies"]), len(self.data["kxs"]))
        except Exception:
            target_shape = None

        def add_map(name, arr, source):
            try:
                Z = np.asarray(arr)
                if target_shape is not None and Z.shape != target_shape:
                    return
                if Z.ndim != 2:
                    return
                # Save real maps as float. If user somehow made a complex custom map,
                # store real/imag separately instead of silently discarding phase.
                if np.iscomplexobj(Z):
                    maps[name + "__real"] = np.real(Z)
                    maps[name + "__imag"] = np.imag(Z)
                    meta.append({"key": name + "__real", "source": source, "complex_part": "real"})
                    meta.append({"key": name + "__imag", "source": source, "complex_part": "imag"})
                else:
                    maps[name] = Z.astype(float, copy=False)
                    meta.append({"key": name, "source": source})
            except Exception:
                return

        # 1) All cached 2D maps.
        for k, v in list(getattr(self, "_cache", {}).items()):
            if isinstance(v, dict):
                # Some old routines cache dictionaries of maps. Save each 2D entry.
                for subk, subv in v.items():
                    add_map(self._cache_key_to_npz_name((k, subk)), subv, f"cache[{repr(k)}][{subk}]")
            else:
                add_map(self._cache_key_to_npz_name(k), v, f"cache[{repr(k)}]")

        # 2) Assigned custom variables v1, v2, ...
        for name, arr in getattr(self, "custom_vars", {}).items():
            add_map("viewer_var__" + self._safe_npz_key_component(name), arr, f"assigned variable {name}")

        # 3) Current map shown on screen.
        if getattr(self, "current_map", None) is not None:
            q = getattr(self, "current_quantity", "current") or "current"
            stamp = time.strftime("%Y%m%d_%H%M%S")
            add_map("viewer_current__" + self._safe_npz_key_component(q) + "__" + stamp, self.current_map, "current displayed map")

        return maps, meta

    def write_computed_maps_to_npz(self):
        """Append viewer-computed 2D maps to the currently loaded NPZ.

        npz files are zip archives, so this is implemented by rewriting the NPZ
        safely to a temporary file, creating a .bak backup, then replacing the
        original file. The raw solver arrays are preserved exactly as loaded.
        """
        if self.data is None or not self.path:
            QMessageBox.information(self, "No NPZ", "Load an NPZ first.")
            return

        maps, map_meta = self._collect_2d_viewer_maps_for_npz()
        if not maps:
            QMessageBox.information(self, "No computed maps", "There are no computed 2D viewer maps to write yet. Plot or assign some signals first.")
            return

        msg = (
            f"This will write {len(maps)} computed 2D map(s) into:\n{self.path}\n\n"
            "A backup with suffix .bak will be created first. Existing viewer_cache/viewer_var keys with the same names will be overwritten. Continue?"
        )
        if QMessageBox.question(self, "Write computed maps to NPZ", msg, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        self.begin_busy("Writing computed maps to NPZ...")
        try:
            QApplication.processEvents()

            # Load every original array before closing/replacing the archive.
            existing = {}
            for k in self.data.files:
                # Drop old metadata only if replacing it below; preserve all arrays.
                existing[k] = self.data[k]

            # Merge metadata history.
            history = []
            if "viewer_saved_maps_json" in existing:
                try:
                    old = existing["viewer_saved_maps_json"]
                    if hasattr(old, "shape"):
                        old_s = str(old.item() if old.shape == () else old.tolist())
                    else:
                        old_s = str(old)
                    parsed = json.loads(old_s)
                    if isinstance(parsed, list):
                        history = parsed
                    elif isinstance(parsed, dict):
                        history = [parsed]
                except Exception:
                    history = []

            entry = {
                "viewer_version": VIEWER_VERSION,
                "saved_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_npz": self.path,
                "map_count": len(maps),
                "maps": map_meta,
                "gui_state": {
                    "incident": self.pol_box.currentText(),
                    "quantity": self.quantity_box.currentText(),
                    "analyzer": self.analyzer_box.currentText(),
                    "region": self.region_box.currentText() if hasattr(self, "region_box") else "Sample",
                    "ky_slice": self.ky_box.currentText(),
                    "order_mode": self.order_mode_box.currentText(),
                    "plot_mode": self.plot_mode_box.currentText(),
                    "cut_E_eV": self.cut_E_edit.text(),
                    "cut_kx_um^-1": self.cut_kx_edit.text(),
                    "custom_expression": self.custom_expr_edit.toPlainText(),
                    "custom_signal_specs": self.custom_var_specs,
                },
            }
            history.append(entry)
            existing["viewer_saved_maps_json"] = np.array(json.dumps(history, indent=2))

            # Add/overwrite computed maps.
            for k, v in maps.items():
                existing[k] = np.asarray(v)

            # Close archive before replacing on Windows.
            try:
                self.data.close()
            except Exception:
                pass

            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            base = os.path.basename(self.path)
            fd, tmp_path = tempfile.mkstemp(prefix=base + ".tmp_", suffix=".npz", dir=directory)
            os.close(fd)
            try:
                np.savez_compressed(tmp_path, **existing)
                backup_path = self.path + ".bak"
                try:
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    shutil.copy2(self.path, backup_path)
                except Exception:
                    # If backup creation fails, do not destroy the original.
                    raise RuntimeError("Could not create backup file. Original NPZ was not modified.")
                os.replace(tmp_path, self.path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # Reload and clear only cache entries that came from old archive handles.
            self.data = np.load(self.path, allow_pickle=True)
            self._cache.clear()
            self.loading_label.setText(f"Wrote {len(maps)} viewer maps into NPZ")
            self.statusBar().showMessage(f"Updated NPZ and created backup: {self.path}.bak", 8000)
            self.update_metadata()
            self.end_busy("Computed maps written to NPZ")
            QMessageBox.information(self, "NPZ updated", f"Wrote {len(maps)} computed map(s) into:\n{self.path}\n\nBackup:\n{self.path}.bak")
        except Exception as e:
            # Try to reopen if we closed before failing.
            try:
                if self.path and (self.data is None or not hasattr(self.data, "files")):
                    self.data = np.load(self.path, allow_pickle=True)
            except Exception:
                pass
            self.end_busy("Write to NPZ failed")
            QMessageBox.critical(self, "Write computed maps error", str(e))

    def save_viewer_config_json(self):
        if self.data is None and not self.path:
            QMessageBox.information(self, "No viewer state", "Load an NPZ or set viewer controls before saving a config.")
            return
        default_name = "rcwa_viewer_config.json"
        try:
            if self.path:
                stem = str(self.path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
                default_name = stem + "_viewer_config.json"
        except Exception:
            pass
        path, _ = QFileDialog.getSaveFileName(self, "Save viewer JSON configuration", default_name, "JSON file (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.viewer_config_dict(), f, indent=2)
            self.statusBar().showMessage(f"Saved viewer config: {path}", 5000)
            QMessageBox.information(self, "Saved viewer configuration", f"Saved viewer configuration to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save viewer config error", str(e))

    def load_viewer_config_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load viewer JSON configuration", "", "JSON file (*.json)")
        if not path:
            return
        self.begin_busy("Loading viewer config...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            npz_path = cfg.get("npz_path")
            # If no NPZ is currently loaded, try to load the one stored in the viewer config.
            if self.data is None and npz_path:
                try:
                    self.open_npz(npz_path)
                except Exception:
                    pass

            controls = cfg.get("controls", {})
            widgets_to_block = [self.pol_box, self.quantity_box, self.analyzer_box, self.region_box,
                                self.ky_box, self.order_mode_box, self.plot_mode_box]
            for w in widgets_to_block:
                try:
                    w.blockSignals(True)
                except Exception:
                    pass
            try:
                self._set_combo_if_present(self.pol_box, controls.get("incident_polarization"))
                self._set_combo_if_present(self.quantity_box, controls.get("quantity"))
                self._set_combo_if_present(self.analyzer_box, controls.get("analyzer"))
                if hasattr(self, "region_box"):
                    self._set_combo_if_present(self.region_box, controls.get("region"))
                self._set_combo_if_present(self.order_mode_box, controls.get("order_mode"))
                self._set_combo_if_present(self.plot_mode_box, controls.get("plot_mode"))
                if self.ky_box.count() > 0:
                    ky_idx = int(controls.get("ky_index", self.ky_box.currentIndex()))
                    self.ky_box.setCurrentIndex(int(np.clip(ky_idx, 0, self.ky_box.count() - 1)))
            finally:
                for w in widgets_to_block:
                    try:
                        w.blockSignals(False)
                    except Exception:
                        pass

            self.cut_E_edit.setText(str(controls.get("cut_E_eV", "")))
            self.cut_kx_edit.setText(str(controls.get("cut_kx_um^-1", "")))
            self.auto_scale_check.setChecked(bool(controls.get("auto_robust_color_scale", True)))
            self.show_dashed_lines_check.setChecked(bool(controls.get("show_dashed_guide_lines", True)))
            self.vmin_edit.setText(str(controls.get("vmin", "")))
            self.vmax_edit.setText(str(controls.get("vmax", "")))
            if hasattr(self, "inc_custom_ratio_edit"):
                self.inc_custom_ratio_edit.setText(str(controls.get("inc_custom_ratio", "1.0")))
                self.inc_custom_phase_edit.setText(str(controls.get("inc_custom_phase_deg", "90.0")))
                self.ana_custom_ratio_edit.setText(str(controls.get("analyzer_custom_ratio", "1.0")))
                self.ana_custom_phase_edit.setText(str(controls.get("analyzer_custom_phase_deg", "90.0")))
            self.update_analyzer_enabled()

            custom = cfg.get("custom", {})
            self.custom_expr_edit.setPlainText(str(custom.get("expression", "")))
            self.clear_custom_signals()
            # Recreate assigned variables from their signal definitions, if an NPZ is loaded.
            if self.data is not None:
                for spec0 in custom.get("signals", []):
                    spec = dict(spec0)
                    name = str(spec.get("name", f"v{self.custom_var_counter + 1}"))
                    Z = self.signal_map(
                        spec.get("incident", "s"),
                        spec.get("quantity", "R"),
                        spec.get("analyzer", "None"),
                        spec.get("order_mode", "zeroth order"),
                        spec.get("region", "Sample"),
                    )
                    self.custom_vars[name] = np.asarray(Z, dtype=float).copy()
                    spec["name"] = name
                    self.custom_var_specs.append(spec)
                    self.custom_signal_list.addItem(self._custom_signal_label(spec))
                    m = re.match(r"v(\d+)$", name)
                    if m:
                        self.custom_var_counter = max(self.custom_var_counter, int(m.group(1)))

            self.end_busy("Viewer config loaded")
            # Restore the plot if possible. Custom expression has priority when it was the active plot.
            if self.data is not None:
                if cfg.get("current_plot", {}).get("is_custom_expression", False) and self.custom_expr_edit.toPlainText().strip():
                    self.plot_custom_expression()
                else:
                    self.plot()
            else:
                self.update_metadata()
        except Exception as e:
            self.end_busy("Viewer config load failed")
            QMessageBox.critical(self, "Load viewer config error", str(e))

    def export_solver_config_json(self):
        """Export the solver configuration embedded in the NPZ.

        The solver can load this file with its Load Config button to reproduce the
        same geometry, material mapping, scan axes, top layers, Au settings, and
        polarization-basis settings.
        """
        if self.data is None:
            QMessageBox.information(self, "No file", "Load an NPZ first.")
            return

        raw = self._npz_json_string("config_json") or self._npz_json_string("metadata_json")
        if not raw:
            QMessageBox.warning(
                self,
                "No configuration found",
                "This NPZ does not contain config_json or metadata_json."
            )
            return

        try:
            cfg = json.loads(raw)
        except Exception as e:
            QMessageBox.critical(self, "Configuration error", f"Could not parse embedded JSON:\n{e}")
            return

        default_name = "rcwa_configuration_from_npz.json"
        try:
            if self.path:
                stem = os.path.splitext(os.path.basename(self.path))[0]
                default_name = stem + "_solver_config.json"
        except Exception:
            pass

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export solver JSON configuration",
            default_name,
            "JSON file (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            QMessageBox.information(
                self,
                "Saved",
                "Exported solver configuration to:\n"
                f"{path}\n\nLoad this in the solver using: Load Config."
            )
        except Exception as e:
            QMessageBox.critical(self, "Save configuration error", str(e))

    def export_csv(self):
        if self.current_map is None:
            QMessageBox.information(self, "No map", "Plot a map first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "rcwa_viewer_map.csv", "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        E = self.data["energies"]
        kx = self.data["kxs"]
        q = self.current_quantity or self.quantity_box.currentText()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_npz", self.path])
            w.writerow(["quantity", q])
            w.writerow(["incident", self.pol_box.currentText()])
            w.writerow(["analyzer", self.analyzer_box.currentText()])
            w.writerow(["energy_eV", "kx_um^-1", q])
            for i, e in enumerate(E):
                for j, kv in enumerate(kx):
                    w.writerow([e, kv, self.current_map[i, j]])
        QMessageBox.information(self, "Saved", f"Saved CSV to:\n{path}")

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "rcwa_viewer_plot.png", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.fig.savefig(path, dpi=200)
        QMessageBox.information(self, "Saved", f"Saved PNG to:\n{path}")


def main():
    app = QApplication(sys.argv)
    win = Viewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
