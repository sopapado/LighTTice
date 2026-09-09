#!/usr/bin/env python3
"""
lighTTice Designer v0.3

A small cell designer for creating patterned-layer JSON files for the RCWA solver.
The exported JSON can be loaded directly by lighTTice Solver.

Coordinate convention:
    - Units: microns
    - Origin: center of the simulation macrocell
    - Domain: [-Lx/2,Lx/2] x [-Ly/2,Ly/2]
    - Shapes usually represent air/void cutouts in a metal background

Supported shapes:
    slit / rectangle: cx, cy, width, height, angle
    ellipse:          cx, cy, rx, ry, angle
    circle:           cx, cy, radius
"""

import sys, json, math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.patches import Rectangle, Ellipse, Circle, Polygon

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QComboBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QSplitter, QGroupBox, QCheckBox
)

APP_VERSION = "0.3-oblique-lattice-periodic-preview"


def fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


class CellDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"lighTTice Designer v{APP_VERSION} — unit-cell editor")
        self.resize(1450, 850)
        self.shapes = []
        self.selected_index = None
        self.current_path = None
        self._updating_table = False
        self._dragging = False
        self._drag_index = None

        self.build_ui()
        self.add_default_slit()
        self.refresh_all()

    def build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        domain_box = QGroupBox("Cell / domain")
        df = QFormLayout(domain_box)
        self.Lx_edit = QLineEdit("0.680")
        self.Ly_edit = QLineEdit("0.680")
        self.Nx_edit = QLineEdit("160")
        self.Ny_edit = QLineEdit("160")
        self.bg_box = QComboBox(); self.bg_box.addItems(["metal", "air"])
        for w in (self.Lx_edit, self.Ly_edit, self.Nx_edit, self.Ny_edit):
            w.setMaximumWidth(95)
            w.editingFinished.connect(self.refresh_plot)
        self.bg_box.currentTextChanged.connect(self.refresh_plot)
        self.lattice_box = QComboBox(); self.lattice_box.addItems(["rectangular", "hexagonal", "custom a1/a2"])
        self.a1x_edit = QLineEdit("0.680"); self.a1y_edit = QLineEdit("0.0")
        self.a2x_edit = QLineEdit("0.0"); self.a2y_edit = QLineEdit("0.680")
        self.period_a_edit = QLineEdit("0.670")
        apply_hex_btn = QPushButton("Apply hex")
        apply_hex_btn.clicked.connect(self.apply_hexagonal_lattice)
        self.lattice_box.currentTextChanged.connect(self.on_lattice_type_changed)
        for w in [self.a1x_edit,self.a1y_edit,self.a2x_edit,self.a2y_edit,self.period_a_edit]:
            w.setMaximumWidth(95)
            w.editingFinished.connect(self.refresh_plot)

        df.addRow("Lx, um", self.Lx_edit)
        df.addRow("Ly, um", self.Ly_edit)
        df.addRow("Nx preview", self.Nx_edit)
        df.addRow("Ny preview", self.Ny_edit)
        df.addRow("Background", self.bg_box)
        df.addRow("Lattice", self.lattice_box)
        df.addRow("Hex a, um", self.period_a_edit)
        df.addRow("a1 x,y", self._row2(self.a1x_edit, self.a1y_edit))
        df.addRow("a2 x,y", self._row2(self.a2x_edit, self.a2y_edit))
        df.addRow("", apply_hex_btn)
        left_layout.addWidget(domain_box)

        shape_box = QGroupBox("Selected / new shape")
        sf = QFormLayout(shape_box)
        self.type_box = QComboBox(); self.type_box.addItems(["slit", "rectangle", "ellipse", "circle"])
        self.material_box = QComboBox(); self.material_box.addItems(["air", "metal"])
        self.name_edit = QLineEdit("shape")
        self.cx_edit = QLineEdit("0.0")
        self.cy_edit = QLineEdit("0.0")
        self.width_edit = QLineEdit("0.050")
        self.height_edit = QLineEdit("0.300")
        self.rx_edit = QLineEdit("0.050")
        self.ry_edit = QLineEdit("0.100")
        self.radius_edit = QLineEdit("0.050")
        self.angle_edit = QLineEdit("0.0")
        for w in [self.cx_edit,self.cy_edit,self.width_edit,self.height_edit,self.rx_edit,self.ry_edit,self.radius_edit,self.angle_edit]:
            w.setMaximumWidth(95)
        sf.addRow("Type", self.type_box)
        sf.addRow("Name", self.name_edit)
        sf.addRow("Material", self.material_box)
        sf.addRow("cx, um", self.cx_edit)
        sf.addRow("cy, um", self.cy_edit)
        sf.addRow("width, um", self.width_edit)
        sf.addRow("height, um", self.height_edit)
        sf.addRow("rx, um", self.rx_edit)
        sf.addRow("ry, um", self.ry_edit)
        sf.addRow("radius, um", self.radius_edit)
        sf.addRow("angle, deg", self.angle_edit)
        left_layout.addWidget(shape_box)

        btns1 = QHBoxLayout()
        add_btn = QPushButton("Add shape")
        upd_btn = QPushButton("Update selected")
        del_btn = QPushButton("Delete")
        dup_btn = QPushButton("Duplicate")
        add_btn.clicked.connect(self.add_shape_from_fields)
        upd_btn.clicked.connect(self.update_selected_from_fields)
        del_btn.clicked.connect(self.delete_selected)
        dup_btn.clicked.connect(self.duplicate_selected)
        for b in [add_btn, upd_btn, del_btn, dup_btn]: btns1.addWidget(b)
        left_layout.addLayout(btns1)

        btns2 = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        example_btn = QPushButton("Example 6 slits")
        honey_btn = QPushButton("Honeycomb holes")
        load_btn = QPushButton("Load JSON")
        save_btn = QPushButton("Save JSON")
        clear_btn.clicked.connect(self.clear_shapes)
        example_btn.clicked.connect(self.make_example_rotated_slits)
        load_btn.clicked.connect(self.load_json)
        save_btn.clicked.connect(self.save_json)
        honey_btn.clicked.connect(self.make_example_honeycomb_holes)
        for b in [clear_btn, example_btn, honey_btn, load_btn, save_btn]: btns2.addWidget(b)
        left_layout.addLayout(btns2)

        self.snap_check = QCheckBox("Click/drag on canvas sets selected shape center")
        self.snap_check.setChecked(True)
        left_layout.addWidget(self.snap_check)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["name", "type", "cx", "cy", "w/rx/r", "h/ry", "angle", "mat"])
        self.table.itemSelectionChanged.connect(self.table_selection_changed)
        left_layout.addWidget(self.table, stretch=1)

        self.status = QLabel("Ready")
        left_layout.addWidget(self.status)

        right = QWidget()
        rl = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([420, 1030])

        self.fig = Figure(figsize=(9.5,7.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        rl.addWidget(self.toolbar)
        rl.addWidget(self.canvas, stretch=1)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def _row2(self, w1, w2):
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(w1); lay.addWidget(w2)
        return box

    def on_lattice_type_changed(self):
        if self.lattice_box.currentText() == "hexagonal":
            self.apply_hexagonal_lattice()
        self.refresh_plot()

    def apply_hexagonal_lattice(self):
        a = fnum(self.period_a_edit.text(), self.Lx())
        self.a1x_edit.setText(f"{a:.9g}"); self.a1y_edit.setText("0")
        self.a2x_edit.setText(f"{0.5*a:.9g}"); self.a2y_edit.setText(f"{math.sqrt(3)/2*a:.9g}")
        # Bounding box of the primitive parallelogram, useful for preview.
        self.Lx_edit.setText(f"{a:.9g}")
        self.Ly_edit.setText(f"{math.sqrt(3)/2*a:.9g}")
        self.refresh_plot()

    def a1(self): return np.array([fnum(self.a1x_edit.text(), self.Lx()), fnum(self.a1y_edit.text(), 0.0)], dtype=float)
    def a2(self): return np.array([fnum(self.a2x_edit.text(), 0.0), fnum(self.a2y_edit.text(), self.Ly())], dtype=float)

    def Lx(self): return fnum(self.Lx_edit.text(), 0.68)
    def Ly(self): return fnum(self.Ly_edit.text(), 0.68)

    def fields_to_shape(self):
        typ = self.type_box.currentText()
        sh = {
            "type": typ,
            "name": self.name_edit.text().strip() or typ,
            "material": self.material_box.currentText(),
            "cx": fnum(self.cx_edit.text()),
            "cy": fnum(self.cy_edit.text()),
            "angle": fnum(self.angle_edit.text()),
        }
        if typ in ("slit", "rectangle"):
            sh["width"] = fnum(self.width_edit.text(), 0.05)
            sh["height"] = fnum(self.height_edit.text(), 0.30)
        elif typ == "ellipse":
            sh["rx"] = fnum(self.rx_edit.text(), 0.05)
            sh["ry"] = fnum(self.ry_edit.text(), 0.10)
        elif typ == "circle":
            sh["radius"] = fnum(self.radius_edit.text(), 0.05)
        return sh

    def shape_to_fields(self, sh):
        typ = sh.get("type", "slit")
        i = self.type_box.findText(typ)
        if i >= 0: self.type_box.setCurrentIndex(i)
        j = self.material_box.findText(sh.get("material", "air"))
        if j >= 0: self.material_box.setCurrentIndex(j)
        self.name_edit.setText(str(sh.get("name", typ)))
        self.cx_edit.setText(str(sh.get("cx", 0.0)))
        self.cy_edit.setText(str(sh.get("cy", 0.0)))
        self.angle_edit.setText(str(sh.get("angle", 0.0)))
        self.width_edit.setText(str(sh.get("width", sh.get("short", 0.05))))
        self.height_edit.setText(str(sh.get("height", sh.get("long", sh.get("length", 0.30)))))
        self.rx_edit.setText(str(sh.get("rx", sh.get("radius_x", 0.05))))
        self.ry_edit.setText(str(sh.get("ry", sh.get("radius_y", 0.10))))
        self.radius_edit.setText(str(sh.get("radius", sh.get("r", 0.05))))

    def add_default_slit(self):
        self.shapes.append({"type":"slit","name":"slit 1","material":"air","cx":0.0,"cy":0.0,"width":0.05,"height":0.30,"angle":0.0})
        self.selected_index = 0

    def add_shape_from_fields(self):
        sh = self.fields_to_shape()
        if sh["name"] == "shape":
            sh["name"] = f"{sh['type']} {len(self.shapes)+1}"
        self.shapes.append(sh)
        self.selected_index = len(self.shapes)-1
        self.refresh_all()

    def update_selected_from_fields(self):
        if self.selected_index is None or not (0 <= self.selected_index < len(self.shapes)):
            QMessageBox.information(self, "No selection", "Select a shape first.")
            return
        self.shapes[self.selected_index] = self.fields_to_shape()
        self.refresh_all()

    def delete_selected(self):
        if self.selected_index is None or not (0 <= self.selected_index < len(self.shapes)):
            return
        self.shapes.pop(self.selected_index)
        self.selected_index = min(self.selected_index, len(self.shapes)-1) if self.shapes else None
        self.refresh_all()

    def duplicate_selected(self):
        if self.selected_index is None or not (0 <= self.selected_index < len(self.shapes)):
            return
        sh = dict(self.shapes[self.selected_index])
        sh["name"] = str(sh.get("name","shape")) + " copy"
        sh["cx"] = fnum(sh.get("cx",0))+0.03
        sh["cy"] = fnum(sh.get("cy",0))+0.03
        self.shapes.append(sh)
        self.selected_index = len(self.shapes)-1
        self.refresh_all()

    def clear_shapes(self):
        self.shapes = []
        self.selected_index = None
        self.refresh_all()

    def make_example_rotated_slits(self):
        self.Lx_edit.setText("4.080")
        self.Ly_edit.setText("0.680")
        self.shapes = []
        Lx, Ly = self.Lx(), self.Ly()
        px = 0.680
        for i in range(6):
            cx = -Lx/2 + (i+0.5)*px
            self.shapes.append({"type":"slit","name":f"slit {i+1}","material":"air","cx":cx,"cy":0.0,"width":0.05,"height":0.30,"angle":30*i})
        self.selected_index = 0
        self.refresh_all()

    def table_selection_changed(self):
        if self._updating_table:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self.selected_index = rows[0].row()
        if 0 <= self.selected_index < len(self.shapes):
            self.shape_to_fields(self.shapes[self.selected_index])
            self.refresh_plot()


    def make_example_honeycomb_holes(self):
        """Two-hole honeycomb basis in a true hexagonal primitive cell."""
        self.lattice_box.setCurrentText("hexagonal")
        self.period_a_edit.setText("0.670")
        self.apply_hexagonal_lattice()
        a = fnum(self.period_a_edit.text(), 0.670)
        self.bg_box.setCurrentText("metal")
        self.shapes = [
            {"type":"circle", "name":"big hole", "material":"air", "cx":0.0, "cy":0.0, "radius":0.150, "angle":0.0},
            {"type":"circle", "name":"small hole", "material":"air", "cx":0.5*a, "cy":math.sqrt(3)/6*a, "radius":0.080, "angle":0.0},
        ]
        self.selected_index = 0
        self.refresh_all()

    def refresh_table(self):
        self._updating_table = True
        self.table.setRowCount(len(self.shapes))
        for r, sh in enumerate(self.shapes):
            typ = sh.get("type", "")
            if typ in ("slit","rectangle"):
                a = sh.get("width", ""); b = sh.get("height", "")
            elif typ == "ellipse":
                a = sh.get("rx", ""); b = sh.get("ry", "")
            else:
                a = sh.get("radius", ""); b = ""
            vals = [sh.get("name", ""), typ, sh.get("cx",0), sh.get("cy",0), a, b, sh.get("angle",0), sh.get("material","air")]
            for c, val in enumerate(vals):
                self.table.setItem(r,c,QTableWidgetItem(str(val)))
        self._updating_table = False
        if self.selected_index is not None and 0 <= self.selected_index < len(self.shapes):
            self.table.selectRow(self.selected_index)
            self.shape_to_fields(self.shapes[self.selected_index])

    def refresh_all(self):
        self.refresh_table()
        self.refresh_plot()

    def draw_shape(self, sh, selected=False, offset=None, ghost=False):
        typ = sh.get("type", "slit")
        off = np.asarray(offset if offset is not None else [0.0, 0.0], dtype=float)
        cx, cy = fnum(sh.get("cx",0)) + off[0], fnum(sh.get("cy",0)) + off[1]
        angle = fnum(sh.get("angle",0))
        face = "white" if sh.get("material","air") == "air" else "gold"
        edge = "red" if selected else ("#555555" if ghost else "black")
        lw = 2.5 if selected else (0.7 if ghost else 1.0)
        alpha = 0.35 if ghost else 1.0
        if typ in ("slit", "rectangle"):
            w, h = fnum(sh.get("width",0.05)), fnum(sh.get("height",0.30))
            patch = Rectangle((cx-w/2, cy-h/2), w, h, angle=angle, facecolor=face, edgecolor=edge, lw=lw, alpha=alpha)
        elif typ == "ellipse":
            patch = Ellipse((cx,cy), 2*fnum(sh.get("rx",0.05)), 2*fnum(sh.get("ry",0.10)), angle=angle, facecolor=face, edgecolor=edge, lw=lw, alpha=alpha)
        elif typ == "circle":
            patch = Circle((cx,cy), fnum(sh.get("radius",0.05)), facecolor=face, edgecolor=edge, lw=lw, alpha=alpha)
        else:
            return
        self.ax.add_patch(patch)
        if not ghost or selected:
            self.ax.plot([cx], [cy], marker="+", color=edge, ms=8)

    def refresh_plot(self):
        self.ax.clear()
        Lx, Ly = self.Lx(), self.Ly()
        a1, a2 = self.a1(), self.a2()
        corners = np.array([-0.5*a1-0.5*a2, 0.5*a1-0.5*a2, 0.5*a1+0.5*a2, -0.5*a1+0.5*a2])
        self.ax.set_facecolor("#d6b44c" if self.bg_box.currentText()=="metal" else "white")
        xmin, xmax = corners[:,0].min(), corners[:,0].max()
        ymin, ymax = corners[:,1].min(), corners[:,1].max()
        pad = 0.10 * max(xmax-xmin, ymax-ymin, 1e-6)
        self.ax.set_xlim(xmin-pad, xmax+pad)
        self.ax.set_ylim(ymin-pad, ymax+pad)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x (um)")
        self.ax.set_ylabel("y (um)")
        self.ax.set_title("Designer cell preview")
        self.ax.grid(True, alpha=0.25)
        self.ax.axhline(0, color="gray", lw=0.7, alpha=0.5)
        self.ax.axvline(0, color="gray", lw=0.7, alpha=0.5)
        self.ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="black", lw=1.7))
        if self.lattice_box.currentText() == "custom a1/a2":
            self.ax.text(corners[0,0], corners[0,1], "cell parallelogram", fontsize=8)
        # Show periodic copies around the primitive cell.  This makes boundary
        # wrapping obvious for hexagonal/oblique lattices and avoids the
        # misleading impression that a boundary-crossing hole is lost.
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                off = ox*a1 + oy*a2
                ghost = not (ox == 0 and oy == 0)
                for i, sh in enumerate(self.shapes):
                    self.draw_shape(sh, selected=((i == self.selected_index) and not ghost), offset=off, ghost=ghost)
        self.ax.text(0.01, 0.01, "black outline = primitive cell; pale copies = periodic images",
                     transform=self.ax.transAxes, fontsize=8, va="bottom", ha="left",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"))
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.status.setText(f"{len(self.shapes)} shapes. a1=({a1[0]:g},{a1[1]:g}), a2=({a2[0]:g},{a2[1]:g}) um")

    def hit_test(self, x, y):
        # simple center-distance hit test
        best, bestd = None, 1e9
        for i, sh in enumerate(self.shapes):
            d = (x - fnum(sh.get("cx",0)))**2 + (y - fnum(sh.get("cy",0)))**2
            if d < bestd:
                bestd = d; best = i
        if best is not None and bestd < (0.08*max(self.Lx(), self.Ly()))**2:
            return best
        return None

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        idx = self.hit_test(event.xdata, event.ydata)
        if idx is not None:
            self.selected_index = idx
            self._dragging = True
            self._drag_index = idx
            self.shapes[idx]["cx"] = float(event.xdata)
            self.shapes[idx]["cy"] = float(event.ydata)
            self.refresh_all()
        elif self.snap_check.isChecked() and self.selected_index is not None and 0 <= self.selected_index < len(self.shapes):
            self.shapes[self.selected_index]["cx"] = float(event.xdata)
            self.shapes[self.selected_index]["cy"] = float(event.ydata)
            self.refresh_all()

    def on_motion(self, event):
        if not self._dragging or self._drag_index is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.shapes[self._drag_index]["cx"] = float(event.xdata)
        self.shapes[self._drag_index]["cy"] = float(event.ydata)
        self.refresh_all()

    def on_release(self, event):
        self._dragging = False
        self._drag_index = None

    def to_dict(self):
        return {
            "format": "lighTTiceCell",
            "version": 1,
            "units": "um",
            "Lx": self.Lx(),
            "Ly": self.Ly(),
            "lattice_type": self.lattice_box.currentText(),
            "a1": [float(self.a1()[0]), float(self.a1()[1])],
            "a2": [float(self.a2()[0]), float(self.a2()[1])],
            "Nx_preview": int(round(fnum(self.Nx_edit.text(),160))),
            "Ny_preview": int(round(fnum(self.Ny_edit.text(),160))),
            "background": self.bg_box.currentText(),
            "shapes": self.shapes,
        }

    def from_dict(self, data):
        self.Lx_edit.setText(str(data.get("Lx", data.get("domain",{}).get("Lx",0.68))))
        self.Ly_edit.setText(str(data.get("Ly", data.get("domain",{}).get("Ly",0.68))))
        lt = data.get("lattice_type", "custom a1/a2" if ("a1" in data and "a2" in data) else "rectangular")
        i_lat = self.lattice_box.findText(lt)
        if i_lat >= 0: self.lattice_box.setCurrentIndex(i_lat)
        a1 = data.get("a1", [data.get("Lx", 0.68), 0.0])
        a2 = data.get("a2", [0.0, data.get("Ly", 0.68)])
        try:
            self.a1x_edit.setText(str(a1[0])); self.a1y_edit.setText(str(a1[1]))
            self.a2x_edit.setText(str(a2[0])); self.a2y_edit.setText(str(a2[1]))
        except Exception:
            pass
        self.Nx_edit.setText(str(data.get("Nx_preview", 160)))
        self.Ny_edit.setText(str(data.get("Ny_preview", 160)))
        bg = data.get("background", "metal")
        i = self.bg_box.findText(bg)
        if i >= 0: self.bg_box.setCurrentIndex(i)
        self.shapes = list(data.get("shapes", []))
        self.selected_index = 0 if self.shapes else None
        self.refresh_all()

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save designer cell", "cell_design.json", "JSON file (*.json)")
        if not path: return
        if not path.lower().endswith(".json"): path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            self.current_path = path
            QMessageBox.information(self, "Saved", f"Saved cell file:\n{path}\n\nLoad this JSON in the solver with 'Load cell JSON'.")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load designer cell", "", "JSON file (*.json);;All files (*)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
            self.current_path = path
            self.status.setText(f"Loaded {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))


def main():
    app = QApplication(sys.argv)
    w = CellDesigner()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
