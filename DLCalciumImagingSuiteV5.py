import sys
import os
import json
import io
import numpy as np
import cv2
import tifffile
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull

# Support for .nd2 files (Nikon NIS-Elements)
try:
    import nd2
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False

# Matplotlib configuration for PyQt6
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# ReportLab for PDF Export
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox, 
    QComboBox, QMessageBox, QLabel, QStackedWidget, QFrame, QMenu, QSlider, QCheckBox,
    QScrollArea, QDialog, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QListWidget, QAbstractItemView, QColorDialog, QSizePolicy, QLineEdit
)
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
import pyqtgraph as pg

CONFIG_FILE = "config.txt"

# ==============================================================================
# AUTHOR & COLLABORATORS CONFIGURATION
# ==============================================================================
LEAD_AUTHOR = "Davide Lattanzi"
CO_AUTHORS = [
    "Stefano Sartini",
    "Ludovica Montesi",
    "Alice Sartini",
    "Giulia Api",
    "Irene Zanchini",
    "Rossana Rauti",
]


# --- WIDGETS WITH WHEEL-EVENT DISABLED ---
class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


# --- DIALOG FOR EXPORTING SINGLE / MULTI-ROI & MULTI-CHANNEL TRACE IMAGE ---
class ExportSingleTraceDialog(QDialog):
    def __init__(self, rois_data, default_roi_idx=0, current_color_hex="#1565c0", parent=None):
        super().__init__(parent)
        self.rois_data = rois_data
        self.setWindowTitle("🖼️ Export Publication Trace Image (Custom ROI & Channel Selection)")
        self.resize(620, 520)
        self.trace_color = QColor(current_color_hex)
        self.overlay_entries = []

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.edit_title = QLineEdit("Fluorescence Trace")
        form.addRow("Plot Title (leave empty for none):", self.edit_title)

        # --- Primary Trace Selection (ROI + Channel + Custom Legend Name + Color) ---
        prim_row = QHBoxLayout()
        self.combo_prim_roi = QComboBox()
        for r in self.rois_data:
            self.combo_prim_roi.addItem(r['name'])
        if 0 <= default_roi_idx < len(self.rois_data):
            self.combo_prim_roi.setCurrentIndex(default_roi_idx)

        self.combo_prim_ch = QComboBox()
        self._populate_channels(self.combo_prim_ch)

        self.edit_prim_label = QLineEdit()
        self._sync_primary_label()
        self.combo_prim_roi.currentIndexChanged.connect(self._sync_primary_label)
        self.combo_prim_ch.currentIndexChanged.connect(self._sync_primary_label)

        self.btn_color = QPushButton("🎨 Color")
        self.btn_color.setStyleSheet(f"background-color: {self.trace_color.name()}; color: white; font-weight: bold; padding: 5px;")
        self.btn_color.clicked.connect(self.pick_primary_color)

        prim_row.addWidget(self.combo_prim_roi)
        prim_row.addWidget(self.combo_prim_ch)
        prim_row.addWidget(self.edit_prim_label, stretch=1)
        prim_row.addWidget(self.btn_color)
        form.addRow("Primary Trace:", prim_row)

        self.combo_scale_style = QComboBox()
        self.combo_scale_style.addItems([
            "Standard X & Y Axes (Ticks + Labels)",
            "Publication L-Scale Bars (X & Y Scale Bars Only)",
            "Both Standard Axes & Grid"
        ])
        form.addRow("X & Y Scale Style:", self.combo_scale_style)

        self.chk_show_peaks = QCheckBox("Highlight Detected Peaks (Channel 1)")
        self.chk_show_peaks.setChecked(True)
        form.addRow("Peaks Overlay:", self.chk_show_peaks)

        self.spin_line_width = QDoubleSpinBox()
        self.spin_line_width.setRange(0.5, 6.0)
        self.spin_line_width.setValue(1.8)
        self.spin_line_width.setSingleStep(0.2)
        form.addRow("Trace Line Width:", self.spin_line_width)

        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(100, 1200)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSingleStep(50)
        form.addRow("Resolution (DPI):", self.spin_dpi)

        layout.addLayout(form)

        # --- Additional Overlaid Traces (Any ROI + Any Channel) ---
        ov_group = QGroupBox("Additional Overlaid Traces (e.g. ROI 1 Ch 1 + ROI 2 Ch 2)")
        ov_layout = QVBoxLayout(ov_group)

        btn_add_ov = QPushButton("➕ Add Overlaid Trace (Select ROI & Channel)")
        btn_add_ov.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold; padding: 5px;")
        btn_add_ov.clicked.connect(self.add_overlay_row)
        ov_layout.addWidget(btn_add_ov)

        self.ov_scroll = QScrollArea()
        self.ov_scroll.setWidgetResizable(True)
        self.ov_container = QWidget()
        self.ov_rows_layout = QVBoxLayout(self.ov_container)
        self.ov_rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ov_scroll.setWidget(self.ov_container)
        ov_layout.addWidget(self.ov_scroll)

        layout.addWidget(ov_group, stretch=1)

        btn_row = QHBoxLayout()
        btn_ok = QPushButton("💾 Export Image (JPEG / TIFF)")
        btn_ok.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;")
        btn_ok.clicked.connect(self.accept)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("padding: 8px;")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _populate_channels(self, combo):
        combo.addItem("Channel 1", 1)
        if self.rois_data and 'ch2' in self.rois_data[0]:
            combo.addItem("Channel 2", 2)
        if self.rois_data and 'ch3' in self.rois_data[0]:
            combo.addItem("Channel 3", 3)

    def _sync_primary_label(self):
        r_name = self.combo_prim_roi.currentText()
        ch_num = self.combo_prim_ch.currentData()
        self.edit_prim_label.setText(f"{r_name} (Ch {ch_num})")

    def pick_primary_color(self):
        c = QColorDialog.getColor(self.trace_color, self, "Select Trace Color")
        if c.isValid():
            self.trace_color = c
            self.btn_color.setStyleSheet(f"background-color: {c.name()}; color: white; font-weight: bold; padding: 5px;")

    def add_overlay_row(self):
        colors_palette = ["#2e7d32", "#d32f2f", "#6a1b9a", "#ef6c00", "#00838f", "#ad1457"]
        row_color = QColor(colors_palette[len(self.overlay_entries) % len(colors_palette)])

        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(2, 2, 2, 2)

        c_roi = QComboBox()
        for r in self.rois_data:
            c_roi.addItem(r['name'])
        c_roi.setCurrentIndex(min(len(self.rois_data) - 1, len(self.overlay_entries) + 1))

        c_ch = QComboBox()
        self._populate_channels(c_ch)
        if c_ch.count() > 1:
            c_ch.setCurrentIndex(1)

        e_lbl = QLineEdit(f"{c_roi.currentText()} (Ch {c_ch.currentData()})")
        c_roi.currentIndexChanged.connect(lambda: e_lbl.setText(f"{c_roi.currentText()} (Ch {c_ch.currentData()})"))
        c_ch.currentIndexChanged.connect(lambda: e_lbl.setText(f"{c_roi.currentText()} (Ch {c_ch.currentData()})"))

        c_style = QComboBox()
        c_style.addItems(["Solid (-)", "Dashed (--)", "Dotted (:)", "Dash-Dot (-.)"])

        entry = {
            'widget': row_w,
            'combo_roi': c_roi,
            'combo_ch': c_ch,
            'edit_lbl': e_lbl,
            'combo_style': c_style,
            'color': row_color
        }

        b_col = QPushButton("🎨")
        b_col.setFixedWidth(34)
        b_col.setStyleSheet(f"background-color: {row_color.name()}; color: white; font-weight: bold;")

        def pick_col():
            c = QColorDialog.getColor(entry['color'], self, "Select Overlaid Trace Color")
            if c.isValid():
                entry['color'] = c
                b_col.setStyleSheet(f"background-color: {c.name()}; color: white; font-weight: bold;")

        b_col.clicked.connect(pick_col)

        b_del = QPushButton("✖")
        b_del.setFixedWidth(28)
        b_del.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")

        def rem_row():
            if entry in self.overlay_entries:
                self.overlay_entries.remove(entry)
            self.ov_rows_layout.removeWidget(row_w)
            row_w.deleteLater()

        b_del.clicked.connect(rem_row)

        row_l.addWidget(c_roi)
        row_l.addWidget(c_ch)
        row_l.addWidget(e_lbl, stretch=1)
        row_l.addWidget(c_style)
        row_l.addWidget(b_col)
        row_l.addWidget(b_del)

        self.overlay_entries.append(entry)
        self.ov_rows_layout.addWidget(row_w)

    def get_traces_to_plot(self):
        style_map = {0: '-', 1: '--', 2: ':', 3: '-.'}
        items = [{
            'roi_idx': self.combo_prim_roi.currentIndex(),
            'channel': self.combo_prim_ch.currentData(),
            'label': self.edit_prim_label.text().strip(),
            'color': self.trace_color.name(),
            'linestyle': '-',
            'is_primary': True
        }]
        for ent in self.overlay_entries:
            items.append({
                'roi_idx': ent['combo_roi'].currentIndex(),
                'channel': ent['combo_ch'].currentData(),
                'label': ent['edit_lbl'].text().strip(),
                'color': ent['color'].name(),
                'linestyle': style_map.get(ent['combo_style'].currentIndex(), '-'),
                'is_primary': False
            })
        return items


# --- DIALOG FOR SELECTING ROIS TO CREATE A CLUSTER ENVELOPE ---
class ClusterSelectionDialog(QDialog):
    def __init__(self, roi_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("✏️ Draw Custom ROI Cluster Envelope")
        self.resize(350, 450)
        self.selected_color = QColor(255, 152, 0, 180)

        layout = QVBoxLayout(self)

        lbl = QLabel("Select the ROIs belonging to the cluster:")
        lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(lbl)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for name in roi_names:
            self.list_widget.addItem(name)
        layout.addWidget(self.list_widget)

        self.btn_color = QPushButton("🎨 Select Contour Color")
        self.btn_color.setStyleSheet(f"background-color: {self.selected_color.name()}; color: white; font-weight: bold; padding: 6px;")
        self.btn_color.clicked.connect(self.choose_color)
        layout.addWidget(self.btn_color)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton("Draw Contour")
        btn_ok.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px;")
        btn_ok.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def choose_color(self):
        col = QColorDialog.getColor(self.selected_color, self, "Select Cluster Contour Color")
        if col.isValid():
            self.selected_color = col
            self.btn_color.setStyleSheet(f"background-color: {col.name()}; color: white; font-weight: bold; padding: 6px;")

    def get_selected_indices(self):
        return [self.list_widget.row(item) for item in self.list_widget.selectedItems()]


# --- COMPREHENSIVE TECHNICAL MANUAL DIALOG ---
class HelpGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        authors_str = LEAD_AUTHOR
        if CO_AUTHORS:
            authors_str += ", " + ", ".join(CO_AUTHORS)
            
        self.setWindowTitle(f"📖 Technical Manual & Data Processing Guide - {authors_str}")
        self.resize(1050, 880)
        
        layout = QVBoxLayout(self)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 10))

        authors_html = f"<b>Lead Author:</b> {LEAD_AUTHOR}"
        if CO_AUTHORS:
            authors_html += f"<br><b>Co-Authors & Collaborators:</b> {', '.join(CO_AUTHORS)}"

        guide_html = f"""
        <div style='font-family: Arial, sans-serif; line-height: 1.6; color: #1e293b;'>
        <h1 style='color: #0f172a; text-align: center; margin-bottom: 2px;'>Calcium Imaging Suite</h1>
        <h3 style='color: #2563eb; text-align: center; margin-top: 0px;'>Comprehensive Technical Manual, User Workflow & Analytical Formulations</h3>
        <p style='text-align: center; font-size: 13px; background-color: #f1f5f9; padding: 8px; border-radius: 6px;'>{authors_html}</p>
        <hr style='border: 1px solid #cbd5e1;'>

        <h2 style='color: #0284c7; border-bottom: 2px solid #0284c7; padding-bottom: 4px;'>PART 1: OPERATIONAL USER WORKFLOW</h2>
        <p>Follow these sequential analytical steps to execute a complete calcium dynamics workflow:</p>
        
        <ol style='line-height: 1.8;'>
            <li><b>Step 1: Time-Series File Loading (Single or Multi-Channel):</b> Click <i>1. Load Stack/Video (Ch 1)</i>. Optionally load a second and third fluorescence channel (<i>+ Load Ch 2</i> / <i>+ Load Ch 3</i>) to inspect simultaneous multi-wavelength dynamics on the exact same cell ROIs.</li>
            <li><b>Step 2: Temporal Axis Calibration:</b> Input the exact sampling interval in milliseconds (<i>Frame Interval in ms</i>). Correct temporal scaling is critical for calculating Rise Time, FWHM, Decay Tau, and AUC in real-world time units.</li>
            <li><b>Step 3: Background ROI Placement (BG):</b> Click <i>Set Background ROI</i> and position the white dashed rectangular box over a non-fluorescent, cell-free region.</li>
            <li><b>Step 4: Neuronal ROI Selection (Circular or Freehand Custom Shape):</b> Click directly on active cell somata to place circular ROIs, or toggle <i>Draw Custom Freehand ROI</i> to trace irregular cell morphologies by dragging with the mouse.</li>
            <li><b>Step 5: Optional Bleach Correction:</b> Choose between <i>Gentle Linear Detrend (Mild)</i>, <i>Exponential Decay Fit (Strong)</i>, <i>Bi-Exponential Decay Fit (Fast + Slow)</i>, or <i>Tri-Exponential Decay Fit (Fast + Mid + Slow)</i> anchored to the initial fluorescence level.</li>
            <li><b>Step 6: Resting Baseline Window & Frame Range Definition:</b> Specify analysis frame bounds and resting-state baseline bounds to compute relative signal changes (dF / F0).</li>
            <li><b>Step 7: Robust Peak Segmentation (MAD Filter):</b> Adjust <i>Noise Threshold (x MAD sigma)</i>, <i>Min Duration</i>, and <i>Min Prominence</i> to segment real biological transients while filtering out baseline noise.</li>
            <li><b>Step 8: Custom ROI & Channel Trace Image Export:</b> Click <i>Export Trace Image (JPEG/TIFF)</i> to select any ROI and Channel combination (e.g. ROI 1 Ch 1 overlaid with ROI 2 Ch 2), customize trace names and colors, and export publication-grade white-background figures.</li>
            <li><b>Step 9: Advanced Transient Kinetics & Reports:</b> Navigate to <i>Advanced Peak Kinetics</i> or export Excel/CSV datasets and PDF Summary Reports.</li>
        </ol>

        <h2 style='color: #0284c7; border-bottom: 2px solid #0284c7; padding-bottom: 4px; margin-top: 25px;'>PART 2: TECHNICAL FEATURES & MATHEMATICAL FORMULATIONS</h2>

        <h3 style='color: #1e40af;'>1. Signal Normalization (dF / F0)</h3>
        <p>Relative fluorescence changes are computed frame-by-frame using the following formula:</p>
        <div style='background-color: #f1f5f9; border-left: 4px solid #2563eb; padding: 10px; margin: 10px 0; font-family: monospace; font-size: 14px;'>
            dF / F0 (t) = [ ( F_raw(t) - F_BG(t) ) - F0 ] / F0
        </div>

        <h3 style='color: #1e40af;'>2. Robust MAD Noise Estimation & Transient Peak Detection</h3>
        <div style='background-color: #f1f5f9; border-left: 4px solid #0284c7; padding: 10px; margin: 10px 0; font-family: monospace; font-size: 14px;'>
            MAD = Median( | X - Median(X) | )<br>
            Sigma_Robust = 1.4826 * MAD<br>
            Noise_Threshold = K * Sigma_Robust
        </div>

        <h3 style='color: #1e40af;'>3. Rising & Decay Kinetics</h3>
        <div style='background-color: #f1f5f9; border-left: 4px solid #16a34a; padding: 10px; margin: 10px 0; font-family: monospace; font-size: 14px;'>
            Linear Rise Model: y(t) = m * t + q<br>
            Mono-Exponential Decay: y(t) = A * exp( -t / Tau ) + C
        </div>

        <h3 style='color: #1e40af;'>4. Network Synchrony & Event Coincidence Analysis</h3>
        <div style='background-color: #f1f5f9; border-left: 4px solid #ea580c; padding: 10px; margin: 10px 0; font-family: monospace; font-size: 14px;'>
            ECR(A, B) = Count( |t_A - t_B| &le; W ) / min( N_A, N_B )
        </div>

        </div>
        """

        text_edit.setHtml(guide_html)
        layout.addWidget(text_edit)

        btn_close = QPushButton("Close Technical Manual")
        btn_close.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; padding: 10px; font-size: 13px;")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


# --- MOUSE CLICKABLE & FREEHAND DRAWABLE IMAGE VIEWBOX ---
class CustomViewBox(pg.ViewBox):
    sig_image_clicked = pyqtSignal(float, float)
    sig_freehand_finished = pyqtSignal(list)

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.freehand_mode = False
        self._drawing = False
        self._current_points = []
        self._temp_curve = None

    def set_freehand_mode(self, enabled):
        self.freehand_mode = enabled
        self._drawing = False
        self._current_points = []
        if self._temp_curve is not None:
            self.removeItem(self._temp_curve)
            self._temp_curve = None

    def mouseClickEvent(self, ev):
        if self.freehand_mode:
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToView(ev.pos())
            self.sig_image_clicked.emit(pos.x(), pos.y())
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if not self.freehand_mode:
            super().mouseDragEvent(ev, axis=axis)
            return

        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        ev.accept()
        pos = self.mapToView(ev.pos())

        if ev.isStart():
            self._drawing = True
            self._current_points = [(pos.x(), pos.y())]
            if self._temp_curve is not None:
                self.removeItem(self._temp_curve)
            self._temp_curve = pg.PlotCurveItem(
                [pos.x()], [pos.y()],
                pen=pg.mkPen('#00e5ff', width=2, style=Qt.PenStyle.DashLine)
            )
            self.addItem(self._temp_curve)

        elif self._drawing:
            self._current_points.append((pos.x(), pos.y()))
            xs = [p[0] for p in self._current_points]
            ys = [p[1] for p in self._current_points]
            if self._temp_curve is not None:
                self._temp_curve.setData(xs, ys)

        if ev.isFinish() and self._drawing:
            self._drawing = False
            if self._temp_curve is not None:
                self.removeItem(self._temp_curve)
                self._temp_curve = None

            if len(self._current_points) >= 3:
                pts = self._current_points
                if len(pts) > 35:
                    step = max(1, len(pts) // 30)
                    pts = pts[::step]
                self.sig_freehand_finished.emit(pts)
            self._current_points = []


class CalciumFluorescenceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Calcium Imaging Suite - {LEAD_AUTHOR}")
        self.resize(1500, 980)

        if os.path.exists("Icona1.ico"):
            self.setWindowIcon(QIcon("Icona1.ico"))

        # Multi-channel stack storage: index 0 = Primary (Ch1), index 1 = Ch2, index 2 = Ch3
        self.stack = None
        self.stack_ch2 = None
        self.stack_ch3 = None
        self.current_stack_path = None
        self.stack_ch2_path = None
        self.stack_ch3_path = None

        self.last_dir = ""  
        self.rois = []
        self.roi_labels = []
        self.roi_custom_colors = {}
        self.bg_roi = None
        self.bg_label = None
        # 0 = None, 1 = Gentle Linear, 2 = Mono-Exp, 3 = Bi-Exp, 4 = Tri-Exp
        self.bleach_corr_mode = 0
        
        self.manual_peaks_override = {}
        self.disabled_peaks = set()
        self.overlay_history = []

        self.peak_cursors_map = {}

        self.v_line_start = None
        self.v_line_end = None
        self.h_line_base = None

        self.central_stacked = QStackedWidget()
        self.setCentralWidget(self.central_stacked)

        self.welcome_widget = self.create_welcome_screen()
        self.main_app_widget = QWidget()
        self.init_main_ui()
        self.peak_analysis_widget = QWidget()
        self.init_peak_analysis_ui()

        self.central_stacked.addWidget(self.welcome_widget)        # Index 0
        self.central_stacked.addWidget(self.main_app_widget)       # Index 1
        self.central_stacked.addWidget(self.peak_analysis_widget) # Index 2

        self.central_stacked.setCurrentIndex(0)

    def create_welcome_screen(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #0b0f19; color: #ffffff;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #0d1322; padding: 12px; border-radius: 10px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)

        tag_label = QLabel("CALCIUM IMAGING SUITE")
        tag_label.setStyleSheet("color: #64b5f6; font-weight: bold; font-size: 11px;")
        left_layout.addWidget(tag_label)

        title = QLabel("Fluorescence Calcium\nAnalyzer")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff;")
        left_layout.addWidget(title)

        subtitle = QLabel("Advanced analytical platform for calcium wave propagation mapping, multi-channel colocalization, event correlation, and transient kinetics.")
        subtitle.setStyleSheet("color: #94a3b8; font-size: 11px;")
        subtitle.setWordWrap(True)
        left_layout.addWidget(subtitle)

        dev_box = QFrame()
        dev_box.setStyleSheet("background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 8px;")
        dev_layout = QVBoxLayout(dev_box)
        dev_layout.setContentsMargins(6, 6, 6, 6)
        dev_layout.setSpacing(2)

        dev_title = QLabel("DEVELOPER & TEAM")
        dev_title.setStyleSheet("color: #64748b; font-weight: bold; font-size: 9px;")
        dev_layout.addWidget(dev_title)

        dev_name = QLabel(f"Lead Author: {LEAD_AUTHOR}")
        dev_name.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        dev_name.setStyleSheet("color: #4fc3f7;")
        dev_layout.addWidget(dev_name)

        if CO_AUTHORS:
            co_authors_label = QLabel("<b>Co-Authors & Collaborators:</b><br>" + "<br>• ".join([""] + CO_AUTHORS))
            co_authors_label.setStyleSheet("color: #cbd5e1; font-size: 10px; margin-top: 2px;")
            co_authors_label.setWordWrap(True)
            dev_layout.addWidget(co_authors_label)

        info_text = QLabel(
            "<br><b>Key Features:</b><br>"
            "• Multi-Channel Support (Up to 3 Simultaneous Fluorescence Channels)<br>"
            "• Circular & Custom Freehand Polygon ROI Drawing<br>"
            "• Multi-Mode Bleach Correction (Linear, 1-Exp, 2-Exp & 3-Exp Fits)<br>"
            "• Continuous Pearson & Discrete Event Coincidence Rate (ECR)<br>"
            "• Top 3 Peak-Coincidence HUB Neurons Identification<br>"
            "• White-Background Publication Trace Export (JPEG/TIFF) & PDF Reports"
        )
        info_text.setStyleSheet("color: #94a3b8; font-size: 10px; line-height: 1.2;")
        info_text.setWordWrap(True)
        dev_layout.addWidget(info_text)

        left_layout.addWidget(dev_box)

        btn_start = QPushButton("🚀 Launch Software")
        btn_start.setStyleSheet("""
            QPushButton {
                background-color: #1565c0; color: white; font-size: 14px; font-weight: bold;
                padding: 10px; border-radius: 6px; border: none; margin-top: 5px;
            }
            QPushButton:hover { background-color: #1e88e5; }
        """)
        btn_start.clicked.connect(lambda: self.central_stacked.setCurrentIndex(1))
        left_layout.addWidget(btn_start)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(8)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.img_label.setStyleSheet("border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px; background-color: #050810;")
        
        self.load_welcome_image()

        btn_change_photo = QPushButton("📷 Select / Change Custom Neurons Photo")
        btn_change_photo.setStyleSheet("""
            QPushButton {
                background-color: #263238; color: #b0bec5; font-size: 11px; font-weight: bold;
                padding: 6px; border-radius: 5px; border: 1px solid #37474f;
            }
            QPushButton:hover { background-color: #37474f; color: white; }
        """)
        btn_change_photo.clicked.connect(self.change_welcome_photo)

        right_panel.addWidget(self.img_label, stretch=1)
        right_panel.addWidget(btn_change_photo)

        layout.addWidget(left_panel, stretch=2)
        layout.addLayout(right_panel, stretch=3)

        return widget

    def load_welcome_image(self):
        saved_path = None
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    saved_path = f.read().strip()
            except Exception:
                pass

        if saved_path and os.path.exists(saved_path):
            pixmap = QPixmap(saved_path)
            if not pixmap.isNull():
                self.img_label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                return

        for img_name in ["MAX_rep30min.jpg", "neuroni.png", "neuroni.jpg", "neuroni.jpeg"]:
            if os.path.exists(img_name):
                pixmap = QPixmap(img_name)
                if not pixmap.isNull():
                    self.img_label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    return

        pixmap = QPixmap(500, 500)
        pixmap.fill(Qt.GlobalColor.black)
        self.img_label.setPixmap(pixmap)

    def change_welcome_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Custom Neurons Photo", self.last_dir, "Images (*.png *.jpg *.jpeg *.tif)")
        if file_path:
            self.last_dir = os.path.dirname(file_path)
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.img_label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                with open(CONFIG_FILE, "w") as f:
                    f.write(file_path)

    # ==========================================
    # MAIN APP UI
    # ==========================================
    def init_main_ui(self):
        main_layout = QHBoxLayout(self.main_app_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(450)
        scroll_area.setMaximumWidth(490)

        panel_widget = QWidget()
        control_panel = QVBoxLayout(panel_widget)
        control_panel.setContentsMargins(6, 6, 6, 6)
        control_panel.setSpacing(8)

        top_btn_layout = QHBoxLayout()
        btn_back = QPushButton("← Welcome Screen")
        btn_back.clicked.connect(lambda: self.central_stacked.setCurrentIndex(0))
        
        btn_help = QPushButton("📖 Full User Guide")
        btn_help.setStyleSheet("background-color: #0288d1; color: white; font-weight: bold;")
        btn_help.clicked.connect(self.open_help_dialog)

        top_btn_layout.addWidget(btn_back)
        top_btn_layout.addWidget(btn_help)
        control_panel.addLayout(top_btn_layout)

        # --- MULTI-CHANNEL STACK LOADING GROUP ---
        load_group = QGroupBox("1. Multi-Channel Stack / Video Loader")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(5)

        self.btn_load = QPushButton("📂 Load Primary Stack (Channel 1)")
        self.btn_load.clicked.connect(lambda: self.load_file())
        self.btn_load.setMinimumHeight(32)
        self.btn_load.setStyleSheet("font-weight: bold; background-color: #1565c0; color: white;")
        load_layout.addWidget(self.btn_load)

        ch_btns_row = QHBoxLayout()
        self.btn_load_ch2 = QPushButton("+ Load Ch 2")
        self.btn_load_ch2.clicked.connect(lambda: self.load_extra_channel(2))
        self.btn_load_ch2.setEnabled(False)
        self.btn_load_ch2.setMinimumHeight(26)
        self.btn_load_ch2.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")

        self.btn_load_ch3 = QPushButton("+ Load Ch 3")
        self.btn_load_ch3.clicked.connect(lambda: self.load_extra_channel(3))
        self.btn_load_ch3.setEnabled(False)
        self.btn_load_ch3.setMinimumHeight(26)
        self.btn_load_ch3.setStyleSheet("background-color: #ad1457; color: white; font-weight: bold;")

        self.btn_clear_extra_ch = QPushButton("🧹 Clear Ch 2/3")
        self.btn_clear_extra_ch.clicked.connect(self.clear_extra_channels)
        self.btn_clear_extra_ch.setEnabled(False)
        self.btn_clear_extra_ch.setMinimumHeight(26)

        ch_btns_row.addWidget(self.btn_load_ch2)
        ch_btns_row.addWidget(self.btn_load_ch3)
        ch_btns_row.addWidget(self.btn_clear_extra_ch)
        load_layout.addLayout(ch_btns_row)

        ch_view_form = QFormLayout()
        self.combo_display_channel = NoWheelComboBox()
        self.combo_display_channel.addItems([
            "Channel 1 (Primary)",
            "Channel 2",
            "Channel 3",
            "RGB Composite (Ch1=G, Ch2=R, Ch3=B)"
        ])
        self.combo_display_channel.setEnabled(False)
        self.combo_display_channel.setMinimumHeight(26)
        self.combo_display_channel.currentIndexChanged.connect(self.refresh_displayed_image)
        ch_view_form.addRow("Image View Channel:", self.combo_display_channel)
        load_layout.addLayout(ch_view_form)

        load_group.setLayout(load_layout)
        control_panel.addWidget(load_group)

        time_group = QGroupBox("Time Axis Calibration")
        time_layout = QFormLayout()

        self.spin_frame_interval_ms = NoWheelDoubleSpinBox()
        self.spin_frame_interval_ms.setRange(0.1, 60000.0)
        self.spin_frame_interval_ms.setValue(100.0)
        self.spin_frame_interval_ms.setDecimals(1)
        self.spin_frame_interval_ms.setSingleStep(5.0)
        self.spin_frame_interval_ms.setMinimumHeight(26)
        self.spin_frame_interval_ms.valueChanged.connect(self.update_all_traces)

        self.combo_time_unit = NoWheelComboBox()
        self.combo_time_unit.addItems(["Milliseconds (ms)", "Seconds (s)", "Frames"])
        self.combo_time_unit.setMinimumHeight(26)
        self.combo_time_unit.currentIndexChanged.connect(self.update_all_traces)

        time_layout.addRow("Frame Interval (ms):", self.spin_frame_interval_ms)
        time_layout.addRow("X-Axis Unit:", self.combo_time_unit)
        time_group.setLayout(time_layout)
        control_panel.addWidget(time_group)

        session_group = QGroupBox("Project Session Management")
        session_layout = QVBoxLayout()
        session_layout.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_save_session = QPushButton("💾 Save Session")
        self.btn_save_session.clicked.connect(self.save_session)
        self.btn_save_session.setEnabled(False)
        self.btn_save_session.setMinimumHeight(28)
        self.btn_save_session.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")

        self.btn_load_session = QPushButton("📂 Open Session")
        self.btn_load_session.clicked.connect(self.load_session)
        self.btn_load_session.setMinimumHeight(28)
        self.btn_load_session.setStyleSheet("background-color: #00695c; color: white; font-weight: bold;")

        btn_row.addWidget(self.btn_save_session)
        btn_row.addWidget(self.btn_load_session)
        session_layout.addLayout(btn_row)

        self.btn_close_session = QPushButton("❌ Close Current Session")
        self.btn_close_session.clicked.connect(self.close_session)
        self.btn_close_session.setEnabled(False)
        self.btn_close_session.setMinimumHeight(28)
        self.btn_close_session.setStyleSheet("background-color: #bf360c; color: white; font-weight: bold;")
        session_layout.addWidget(self.btn_close_session)

        session_group.setLayout(session_layout)
        control_panel.addWidget(session_group)

        frame_group = QGroupBox("Frame Navigation")
        frame_layout = QVBoxLayout()
        frame_layout.setSpacing(4)
        self.lbl_frame_info = QLabel("Frame: 0 / 0 (Max Projection)")
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self.on_frame_slider_changed)

        self.btn_show_max_proj = QPushButton("🔍 Show Max Intensity Projection")
        self.btn_show_max_proj.setEnabled(False)
        self.btn_show_max_proj.clicked.connect(self.show_max_projection)

        frame_layout.addWidget(self.lbl_frame_info)
        frame_layout.addWidget(self.slider_frame)
        frame_layout.addWidget(self.btn_show_max_proj)
        frame_group.setLayout(frame_layout)
        control_panel.addWidget(frame_group)

        roi_group = QGroupBox("ROI Management")
        roi_layout = QVBoxLayout()
        roi_layout.setSpacing(5)
        self.btn_set_bg = QPushButton("Set Background ROI (Gray)")
        self.btn_set_bg.clicked.connect(self.set_bg_roi)
        self.btn_set_bg.setEnabled(False)
        self.btn_set_bg.setMinimumHeight(28)
        roi_layout.addWidget(self.btn_set_bg)

        self.btn_add_roi = QPushButton("Add Circular ROI at Center")
        self.btn_add_roi.clicked.connect(lambda: self.add_roi())
        self.btn_add_roi.setEnabled(False)
        self.btn_add_roi.setMinimumHeight(28)
        roi_layout.addWidget(self.btn_add_roi)

        self.btn_freehand_roi = QPushButton("✍️ Draw Custom Freehand ROI (Off)")
        self.btn_freehand_roi.setCheckable(True)
        self.btn_freehand_roi.clicked.connect(self.toggle_freehand_roi_mode)
        self.btn_freehand_roi.setEnabled(False)
        self.btn_freehand_roi.setMinimumHeight(28)
        self.btn_freehand_roi.setStyleSheet("background-color: #00838f; color: white; font-weight: bold;")
        roi_layout.addWidget(self.btn_freehand_roi)

        self.btn_delete_selected_roi = QPushButton("🗑️ Delete Selected ROI")
        self.btn_delete_selected_roi.clicked.connect(self.delete_selected_roi)
        self.btn_delete_selected_roi.setEnabled(False)
        self.btn_delete_selected_roi.setMinimumHeight(28)
        self.btn_delete_selected_roi.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        roi_layout.addWidget(self.btn_delete_selected_roi)
        roi_group.setLayout(roi_layout)
        control_panel.addWidget(roi_group)

        bleach_group = QGroupBox("Bleach Correction")
        bleach_layout = QFormLayout()
        self.combo_bleach_mode = NoWheelComboBox()
        self.combo_bleach_mode.addItems([
            "None (Off)",
            "⚡ Gentle Linear Detrend (Mild)",
            "⚡ Exponential Decay Fit (1-Exp)",
            "⚡ Bi-Exponential Decay Fit (2-Exp: Fast + Slow)",
            "⚡ Tri-Exponential Decay Fit (3-Exp: Fast + Mid + Slow)"
        ])
        self.combo_bleach_mode.setEnabled(False)
        self.combo_bleach_mode.setMinimumHeight(28)
        self.combo_bleach_mode.currentIndexChanged.connect(self.on_bleach_mode_changed)
        bleach_layout.addRow("Correction Mode:", self.combo_bleach_mode)
        bleach_group.setLayout(bleach_layout)
        control_panel.addWidget(bleach_group)

        f0_group = QGroupBox("Baseline Window (F0) & Frame Range")
        f0_layout = QFormLayout()
        
        self.spin_analysis_start = NoWheelSpinBox()
        self.spin_analysis_start.setRange(1, 10000)
        self.spin_analysis_start.setValue(1)
        self.spin_analysis_start.setMinimumHeight(26)
        self.spin_analysis_start.valueChanged.connect(self.update_all_traces)

        self.spin_analysis_end = NoWheelSpinBox()
        self.spin_analysis_end.setRange(1, 10000)
        self.spin_analysis_end.setValue(100)
        self.spin_analysis_end.setMinimumHeight(26)
        self.spin_analysis_end.valueChanged.connect(self.update_all_traces)

        self.spin_f0_start = NoWheelSpinBox()
        self.spin_f0_start.setRange(1, 10000)
        self.spin_f0_start.setValue(1)
        self.spin_f0_start.setMinimumHeight(26)
        self.spin_f0_start.valueChanged.connect(self.update_all_traces)

        self.spin_f0_end = NoWheelSpinBox()
        self.spin_f0_end.setRange(1, 10000)
        self.spin_f0_end.setValue(20)
        self.spin_f0_end.setMinimumHeight(26)
        self.spin_f0_end.valueChanged.connect(self.update_all_traces)

        f0_layout.addRow("Analysis Start Frame:", self.spin_analysis_start)
        f0_layout.addRow("Analysis End Frame:", self.spin_analysis_end)
        f0_layout.addRow("F0 Start Frame:", self.spin_f0_start)
        f0_layout.addRow("F0 End Frame:", self.spin_f0_end)
        f0_group.setLayout(f0_layout)
        control_panel.addWidget(f0_group)

        peak_group = QGroupBox("Robust Peak Detection (MAD Filter)")
        peak_layout = QFormLayout()
        peak_layout.setVerticalSpacing(6)

        self.spin_noise_mult = NoWheelDoubleSpinBox()
        self.spin_noise_mult.setRange(0.05, 20.0)
        self.spin_noise_mult.setValue(0.5)
        self.spin_noise_mult.setSingleStep(0.1)
        self.spin_noise_mult.setDecimals(2)
        self.spin_noise_mult.setMinimumHeight(26)
        self.spin_noise_mult.valueChanged.connect(self.update_all_traces)

        self.spin_min_duration = NoWheelSpinBox()
        self.spin_min_duration.setRange(1, 100)
        self.spin_min_duration.setValue(1)
        self.spin_min_duration.setMinimumHeight(26)
        self.spin_min_duration.valueChanged.connect(self.update_all_traces)

        self.spin_prominence = NoWheelDoubleSpinBox()
        self.spin_prominence.setRange(0.00001, 10000.0)
        self.spin_prominence.setValue(0.0001)
        self.spin_prominence.setDecimals(5)
        self.spin_prominence.setSingleStep(0.0005)
        self.spin_prominence.setMinimumHeight(26)
        self.spin_prominence.valueChanged.connect(self.update_all_traces)

        self.spin_distance = NoWheelSpinBox()
        self.spin_distance.setRange(1, 1000)
        self.spin_distance.setValue(1)
        self.spin_distance.setMinimumHeight(26)
        self.spin_distance.valueChanged.connect(self.update_all_traces)

        peak_layout.addRow("Noise Threshold (x MAD sigma):", self.spin_noise_mult)
        peak_layout.addRow("Min Peak Duration (Frames):", self.spin_min_duration)
        peak_layout.addRow("Min Prominence:", self.spin_prominence)
        peak_layout.addRow("Min Distance (Frames):", self.spin_distance)
        peak_group.setLayout(peak_layout)
        control_panel.addWidget(peak_group)

        self.btn_main_pdf_report = QPushButton("📄 Generate Final Summary PDF Report")
        self.btn_main_pdf_report.setMinimumHeight(32)
        self.btn_main_pdf_report.setEnabled(False)
        self.btn_main_pdf_report.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 12px;")
        self.btn_main_pdf_report.clicked.connect(self.generate_pdf_report)
        control_panel.addWidget(self.btn_main_pdf_report)

        self.btn_go_peak_analysis = QPushButton("📈 Advanced Peak Kinetics Analysis →")
        self.btn_go_peak_analysis.setEnabled(False)
        self.btn_go_peak_analysis.setMinimumHeight(32)
        self.btn_go_peak_analysis.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 13px;")
        self.btn_go_peak_analysis.clicked.connect(self.open_peak_analysis_page)
        control_panel.addWidget(self.btn_go_peak_analysis)

        disp_group = QGroupBox("Plot Display & Single Trace Export")
        disp_layout = QFormLayout()
        
        self.combo_mode = NoWheelComboBox()
        self.combo_mode.addItems(["ΔF / F0 (Relative)", "Raw - Background (Absolute)", "Raw Without Background"])
        self.combo_mode.setMinimumHeight(26)
        self.combo_mode.currentIndexChanged.connect(self.update_all_traces)
        disp_layout.addRow("Signal Mode:", self.combo_mode)

        self.combo_roi_filter = NoWheelComboBox()
        self.combo_roi_filter.addItem("All ROIs")
        self.combo_roi_filter.setMinimumHeight(26)
        self.combo_roi_filter.currentIndexChanged.connect(self.on_active_trace_filter_changed)
        disp_layout.addRow("Active Trace:", self.combo_roi_filter)

        self.chk_overlay_multichannel = QCheckBox("Overlay Ch 2 & Ch 3 Traces on Plot")
        self.chk_overlay_multichannel.setChecked(True)
        self.chk_overlay_multichannel.setEnabled(False)
        self.chk_overlay_multichannel.toggled.connect(self.update_all_traces)
        disp_layout.addRow(self.chk_overlay_multichannel)

        self.btn_change_trace_color = QPushButton("🎨 Change Selected ROI Trace Color")
        self.btn_change_trace_color.setEnabled(False)
        self.btn_change_trace_color.setMinimumHeight(26)
        self.btn_change_trace_color.clicked.connect(self.change_selected_roi_trace_color)
        disp_layout.addRow(self.btn_change_trace_color)

        self.btn_export_trace_img = QPushButton("🖼️ Export Trace Image (JPEG/TIFF)")
        self.btn_export_trace_img.setEnabled(False)
        self.btn_export_trace_img.setMinimumHeight(28)
        self.btn_export_trace_img.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        self.btn_export_trace_img.clicked.connect(self.export_single_trace_image)
        disp_layout.addRow(self.btn_export_trace_img)

        disp_group.setLayout(disp_layout)
        control_panel.addWidget(disp_group)

        sync_group = QGroupBox("Network Analysis & Spatial Mapping")
        sync_layout = QFormLayout()

        self.combo_cmap = NoWheelComboBox()
        self.combo_cmap.addItems(["coolwarm", "viridis", "plasma", "seismic", "magma", "Spectral"])
        self.combo_cmap.setMinimumHeight(26)
        sync_layout.addRow("Heatmap Palette:", self.combo_cmap)

        self.spin_coincidence_window = NoWheelSpinBox()
        self.spin_coincidence_window.setRange(0, 50)
        self.spin_coincidence_window.setValue(7)
        self.spin_coincidence_window.setSuffix(" frames")
        self.spin_coincidence_window.setMinimumHeight(26)
        sync_layout.addRow("Peak Window (±Frames):", self.spin_coincidence_window)

        self.btn_sync = QPushButton("🧠 Continuous Synchrony Map")
        self.btn_sync.clicked.connect(self.plot_sync_map)
        self.btn_sync.setEnabled(False)
        self.btn_sync.setMinimumHeight(28)
        self.btn_sync.setStyleSheet("font-weight: bold; background-color: #1565c0; color: white;")
        sync_layout.addRow(self.btn_sync)

        self.btn_ecr_sync = QPushButton("⚡ Event Coincidence Rate (ECR)")
        self.btn_ecr_sync.clicked.connect(self.plot_ecr_sync_map)
        self.btn_ecr_sync.setEnabled(False)
        self.btn_ecr_sync.setMinimumHeight(28)
        self.btn_ecr_sync.setStyleSheet("font-weight: bold; background-color: #e65100; color: white;")
        sync_layout.addRow(self.btn_ecr_sync)

        self.btn_event_sync = QPushButton("⚡ Event Peak Correlation (Discrete Jaccard)")
        self.btn_event_sync.clicked.connect(self.plot_event_sync_map)
        self.btn_event_sync.setEnabled(False)
        self.btn_event_sync.setMinimumHeight(28)
        self.btn_event_sync.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
        sync_layout.addRow(self.btn_event_sync)

        self.btn_latency = QPushButton("⏱️ Activation Latency Plot")
        self.btn_latency.clicked.connect(self.plot_activation_sequence)
        self.btn_latency.setEnabled(False)
        self.btn_latency.setMinimumHeight(28)
        self.btn_latency.setStyleSheet("font-weight: bold; background-color: #8e24aa; color: white;")
        sync_layout.addRow(self.btn_latency)

        self.btn_show_hub = QPushButton("👑 Highlight Top HUB Neurons (Peak-Based)")
        self.btn_show_hub.clicked.connect(self.highlight_hub_neurons_peak_based)
        self.btn_show_hub.setEnabled(False)
        self.btn_show_hub.setMinimumHeight(28)
        self.btn_show_hub.setStyleSheet("font-weight: bold; background-color: #f57f17; color: white;")
        sync_layout.addRow(self.btn_show_hub)

        self.btn_draw_cluster = QPushButton("✏️ Create ROI Cluster Envelope")
        self.btn_draw_cluster.clicked.connect(self.draw_roi_cluster_envelope)
        self.btn_draw_cluster.setEnabled(False)
        self.btn_draw_cluster.setMinimumHeight(28)
        self.btn_draw_cluster.setStyleSheet("font-weight: bold; background-color: #e65100; color: white;")
        sync_layout.addRow(self.btn_draw_cluster)

        overlay_ctrl_layout = QHBoxLayout()
        self.btn_undo_overlay = QPushButton("↩️ Undo Overlay")
        self.btn_undo_overlay.clicked.connect(self.undo_last_overlay)
        self.btn_undo_overlay.setEnabled(False)
        self.btn_undo_overlay.setStyleSheet("font-weight: bold; background-color: #455a64; color: white;")

        self.btn_clear_overlay = QPushButton("🧹 Clear Overlay")
        self.btn_clear_overlay.clicked.connect(self.clear_all_overlays)
        self.btn_clear_overlay.setEnabled(False)
        self.btn_clear_overlay.setStyleSheet("font-weight: bold; background-color: #d32f2f; color: white;")

        overlay_ctrl_layout.addWidget(self.btn_undo_overlay)
        overlay_ctrl_layout.addWidget(self.btn_clear_overlay)
        sync_layout.addRow(overlay_ctrl_layout)

        sync_group.setLayout(sync_layout)
        control_panel.addWidget(sync_group)

        self.btn_export = QPushButton("💾 Export General Data (Excel / CSV)")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        self.btn_export.setMinimumHeight(32)
        self.btn_export.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
        control_panel.addWidget(self.btn_export)

        control_panel.addStretch()
        scroll_area.setWidget(panel_widget)
        main_layout.addWidget(scroll_area, stretch=0)

        display_layout = QVBoxLayout()
        
        self.lbl_loaded_filename = QLabel("📁 Active File: No File Loaded")
        self.lbl_loaded_filename.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.lbl_loaded_filename.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 6px; border-radius: 4px; border: 1px solid #334155;")
        display_layout.addWidget(self.lbl_loaded_filename)

        self.image_widget = pg.GraphicsLayoutWidget()
        self.view_box = CustomViewBox()
        self.view_box.setAspectLocked(True)
        self.view_box.sig_image_clicked.connect(self.on_image_clicked)
        self.view_box.sig_freehand_finished.connect(self.on_freehand_roi_finished)
        self.image_widget.addItem(self.view_box)

        self.img_item = pg.ImageItem()
        self.view_box.addItem(self.img_item)
        display_layout.addWidget(self.image_widget, stretch=1)

        self.plot_widget = pg.PlotWidget(title="Fluorescence Traces (Shift+Left Click: +Peak | Shift+Right Click: -Peak)")
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setLabel('left', 'Signal')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)
        display_layout.addWidget(self.plot_widget, stretch=1)

        main_layout.addLayout(display_layout, stretch=1)

    # ==========================================
    # MULTI-CHANNEL LOADING & DISPLAY HELPERS
    # ==========================================
    def _read_stack_from_path(self, file_path):
        if file_path.lower().endswith('.nd2'):
            if not ND2_AVAILABLE:
                QMessageBox.critical(self, "Missing Library", "The 'nd2' Python library is not installed.")
                return None
            try:
                with nd2.ND2File(file_path) as ndfile:
                    raw_data = ndfile.asarray()
                    while raw_data.ndim > 3:
                        raw_data = raw_data[0]
                    return raw_data
            except Exception as e:
                QMessageBox.critical(self, "ND2 Loading Error", f"Unable to read ND2 file:\n{str(e)}")
                return None
        elif file_path.lower().endswith(('.tif', '.tiff')):
            return tifffile.imread(file_path)
        else:
            cap = cv2.VideoCapture(file_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            cap.release()
            return np.array(frames)

    def load_extra_channel(self, channel_num, file_path=None):
        if self.stack is None:
            QMessageBox.warning(self, "Primary Stack Required", "Please load Channel 1 (Primary Stack) first.")
            return False

        if not file_path:
            file_filter = "Compatible Files (*.tif *.tiff *.nd2 *.avi *.mp4);;Image Stacks (*.tif *.tiff *.nd2);;Video Files (*.avi *.mp4);;All Files (*)"
            file_path, _ = QFileDialog.getOpenFileName(self, f"Open Channel {channel_num} Stack/Video", self.last_dir, file_filter)

        if not file_path or not os.path.exists(file_path):
            return False

        self.last_dir = os.path.dirname(file_path)
        loaded_stack = self._read_stack_from_path(file_path)

        if loaded_stack is None or loaded_stack.ndim != 3:
            QMessageBox.critical(self, "Error", f"Invalid 3D time-series stack for Channel {channel_num}.")
            return False

        if loaded_stack.shape[1:] != self.stack.shape[1:]:
            QMessageBox.warning(
                self, "Dimension Mismatch",
                f"Channel {channel_num} spatial resolution {loaded_stack.shape[1:]} differs from Channel 1 {self.stack.shape[1:]}.\n"
                "Please ensure both channels share the same pixel dimensions."
            )
            return False

        if channel_num == 2:
            self.stack_ch2 = loaded_stack
            self.stack_ch2_path = file_path
        elif channel_num == 3:
            self.stack_ch3 = loaded_stack
            self.stack_ch3_path = file_path

        self.chk_overlay_multichannel.setEnabled(True)
        self.btn_clear_extra_ch.setEnabled(True)
        self.update_active_filename_banner()
        self.refresh_displayed_image()
        self.update_all_traces()
        return True

    def clear_extra_channels(self):
        self.stack_ch2 = None
        self.stack_ch3 = None
        self.stack_ch2_path = None
        self.stack_ch3_path = None
        self.chk_overlay_multichannel.setEnabled(False)
        self.btn_clear_extra_ch.setEnabled(False)
        self.combo_display_channel.setCurrentIndex(0)
        self.update_active_filename_banner()
        self.refresh_displayed_image()
        self.update_all_traces()

    def update_active_filename_banner(self):
        parts = []
        if self.current_stack_path:
            parts.append(f"Ch1: {os.path.basename(self.current_stack_path)}")
        if self.stack_ch2_path:
            parts.append(f"Ch2: {os.path.basename(self.stack_ch2_path)}")
        if self.stack_ch3_path:
            parts.append(f"Ch3: {os.path.basename(self.stack_ch3_path)}")

        if parts:
            self.lbl_loaded_filename.setText("📁 Active Channels: " + "  |  ".join(parts))
        else:
            self.lbl_loaded_filename.setText("📁 Active File: No File Loaded")

    def _normalize_2d_to_uint8(self, img2d):
        arr = img2d.astype(np.float32)
        mn, mx = np.min(arr), np.max(arr)
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.uint8)

    def refresh_displayed_image(self, frame_idx=None):
        if self.stack is None:
            return

        ch_mode = self.combo_display_channel.currentIndex()
        use_max = (frame_idx is None)

        def get_slice(stk, idx):
            if stk is None:
                return None
            if use_max:
                return np.max(stk, axis=0)
            safe_idx = min(idx, len(stk) - 1)
            return stk[safe_idx]

        if ch_mode == 0:
            img = get_slice(self.stack, frame_idx)
            self.img_item.setImage(img, axisOrder='row-major')
        elif ch_mode == 1:
            if self.stack_ch2 is not None:
                img = get_slice(self.stack_ch2, frame_idx)
                self.img_item.setImage(img, axisOrder='row-major')
            else:
                QMessageBox.information(self, "Channel 2 Not Loaded", "Channel 2 is not loaded yet.")
                self.combo_display_channel.setCurrentIndex(0)
        elif ch_mode == 2:
            if self.stack_ch3 is not None:
                img = get_slice(self.stack_ch3, frame_idx)
                self.img_item.setImage(img, axisOrder='row-major')
            else:
                QMessageBox.information(self, "Channel 3 Not Loaded", "Channel 3 is not loaded yet.")
                self.combo_display_channel.setCurrentIndex(0)
        elif ch_mode == 3:
            s1 = get_slice(self.stack, frame_idx)
            h, w = s1.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 1] = self._normalize_2d_to_uint8(s1)
            if self.stack_ch2 is not None:
                rgb[:, :, 0] = self._normalize_2d_to_uint8(get_slice(self.stack_ch2, frame_idx))
            if self.stack_ch3 is not None:
                rgb[:, :, 2] = self._normalize_2d_to_uint8(get_slice(self.stack_ch3, frame_idx))
            self.img_item.setImage(rgb, axisOrder='row-major')

    def show_max_projection(self):
        if self.stack is not None:
            self.lbl_frame_info.setText(f"Frame: Max Projection (1..{len(self.stack)})")
            self.refresh_displayed_image(frame_idx=None)

    # ==========================================
    # FREEHAND CUSTOM SHAPE ROI DRAWING
    # ==========================================
    def toggle_freehand_roi_mode(self, checked):
        self.view_box.set_freehand_mode(checked)
        if checked:
            self.btn_freehand_roi.setText("✍️ Drawing Custom ROI: ON (Drag Mouse)")
            self.btn_freehand_roi.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")
        else:
            self.btn_freehand_roi.setText("✍️ Draw Custom Freehand ROI (Off)")
            self.btn_freehand_roi.setStyleSheet("background-color: #00838f; color: white; font-weight: bold;")

    def on_freehand_roi_finished(self, points_list):
        if self.stack is None or len(points_list) < 3:
            return
        self.add_freehand_roi(points_list)
        self.btn_freehand_roi.setChecked(False)
        self.toggle_freehand_roi_mode(False)

    def add_freehand_roi(self, points_list):
        colors_list = ['r', 'g', 'b', 'c', 'm', 'y']
        color_code = colors_list[len(self.rois) % len(colors_list)]
        roi_num = len(self.rois) + 1

        poly_roi = pg.PolyLineROI(points_list, closed=True, pen=pg.mkPen(color_code, width=2))
        poly_roi.sigClicked.connect(self.on_roi_clicked)
        self.view_box.addItem(poly_roi)
        self.rois.append(poly_roi)

        roi_label = pg.TextItem(f"ROI {roi_num}", color=color_code, anchor=(0.5, 0.5))
        roi_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.view_box.addItem(roi_label)
        self.roi_labels.append(roi_label)

        poly_roi.sigRegionChanged.connect(self.update_all_traces)

        self.combo_roi_filter.blockSignals(True)
        self.combo_roi_filter.addItem(f"ROI {roi_num}")
        self.combo_roi_filter.blockSignals(False)

        self.btn_delete_selected_roi.setEnabled(True)
        self.update_all_traces()

    def _get_roi_center(self, roi):
        if isinstance(roi, pg.PolyLineROI):
            pts = [roi.mapToParent(h['item'].pos()) for h in roi.handles]
            if pts:
                cx = np.mean([p.x() for p in pts])
                cy = np.mean([p.y() for p in pts])
                return cx, cy
        pos = roi.pos()
        size = roi.size()
        return pos.x() + size.x() / 2.0, pos.y() + size.y() / 2.0

    # ==========================================
    # SINGLE / MULTI-ROI & CHANNEL TRACE COLOR & IMAGE EXPORT
    # ==========================================
    def on_active_trace_filter_changed(self):
        filter_idx = self.combo_roi_filter.currentIndex()
        is_single_roi = (filter_idx > 0)
        self.btn_change_trace_color.setEnabled(is_single_roi)
        self.btn_export_trace_img.setEnabled(len(self.rois) > 0)
        self.update_all_traces()

    def _get_roi_display_color(self, roi_idx):
        if roi_idx in self.roi_custom_colors:
            return self.roi_custom_colors[roi_idx]
        default_colors = ['#ff1744', '#00e676', '#2979ff', '#00e5ff', '#d500f9', '#ffea00']
        return default_colors[roi_idx % len(default_colors)]

    def change_selected_roi_trace_color(self):
        filter_idx = self.combo_roi_filter.currentIndex()
        if filter_idx <= 0:
            return
        roi_idx = filter_idx - 1
        current_col = QColor(self._get_roi_display_color(roi_idx))
        chosen = QColorDialog.getColor(current_col, self, f"Choose Trace Color for ROI {roi_idx + 1}")
        if chosen.isValid():
            self.roi_custom_colors[roi_idx] = chosen.name()
            if roi_idx < len(self.rois):
                self.rois[roi_idx].setPen(pg.mkPen(chosen.name(), width=2))
            if roi_idx < len(self.roi_labels):
                self.roi_labels[roi_idx].setColor(chosen.name())
            self.update_all_traces()

    def _extract_trace_for_roi_channel(self, roi_info, channel_num, mode_index):
        if channel_num == 1:
            src = roi_info
        elif channel_num == 2 and 'ch2' in roi_info:
            src = roi_info['ch2']
        elif channel_num == 3 and 'ch3' in roi_info:
            src = roi_info['ch3']
        else:
            return None

        if mode_index == 0:
            return src['df_f0']
        elif mode_index == 1:
            return src['sub_bg']
        else:
            return src['raw']

    def export_single_trace_image(self):
        data = self.calculate_data()
        if not data or not data['rois_data']:
            QMessageBox.warning(self, "No ROIs", "Please create at least one ROI first.")
            return

        filter_idx = self.combo_roi_filter.currentIndex()
        default_roi_idx = (filter_idx - 1) if filter_idx > 0 else 0

        dlg = ExportSingleTraceDialog(
            rois_data=data['rois_data'],
            default_roi_idx=default_roi_idx,
            current_color_hex=self._get_roi_display_color(default_roi_idx),
            parent=self
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        traces_to_plot = dlg.get_traces_to_plot()
        prim_cfg = traces_to_plot[0]
        prim_roi_info = data['rois_data'][prim_cfg['roi_idx']]

        default_name = f"{prim_roi_info['name'].replace(' ', '_')}_Ch{prim_cfg['channel']}_trace.tif"
        default_path = os.path.join(self.last_dir, default_name) if self.last_dir else default_name
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trace Image",
            default_path,
            "TIFF Image (*.tif *.tiff);;JPEG Image (*.jpg *.jpeg)"
        )
        if not file_path:
            return

        self.last_dir = os.path.dirname(file_path)

        mode_index = self.combo_mode.currentIndex()
        if mode_index == 0:
            y_label = "ΔF / F0"
        elif mode_index == 1:
            y_label = "Fluorescence (RAW - BG)"
        else:
            y_label = "RAW Fluorescence"

        fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
        ax.set_facecolor('white')

        lw = dlg.spin_line_width.value()
        x_prim, x_label, y_prim = None, "Time", None

        for cfg in traces_to_plot:
            r_idx = cfg['roi_idx']
            ch_num = cfg['channel']
            r_info = data['rois_data'][r_idx]
            y_vals = self._extract_trace_for_roi_channel(r_info, ch_num, mode_index)
            if y_vals is None:
                continue

            x_vals, cur_x_label = self.get_time_axis(len(y_vals))
            lbl_str = cfg['label'] if cfg['label'] else None
            ax.plot(
                x_vals, y_vals,
                color=cfg['color'],
                linewidth=lw if cfg['is_primary'] else lw * 0.95,
                linestyle=cfg['linestyle'],
                label=lbl_str
            )

            if cfg['is_primary']:
                x_prim = x_vals
                x_label = cur_x_label
                y_prim = y_vals

            if dlg.chk_show_peaks.isChecked() and ch_num == 1 and len(r_info['peaks_indices']) > 0:
                p_idx = r_info['peaks_indices']
                ax.scatter(x_vals[p_idx], y_vals[p_idx], color=cfg['color'], edgecolors='black', zorder=5, s=35)

        title_str = dlg.edit_title.text().strip()
        if title_str:
            ax.set_title(title_str, fontsize=12, fontweight='bold', color='black', pad=10)

        scale_style = dlg.combo_scale_style.currentIndex()
        if scale_style in (0, 2):
            ax.set_xlabel(x_label, fontsize=11, fontweight='bold', color='black')
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold', color='black')
            ax.tick_params(colors='black', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            if scale_style == 2:
                ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        else:
            ax.axis('off')
            if x_prim is not None and y_prim is not None:
                x_min, x_max = np.min(x_prim), np.max(x_prim)
                y_min, y_max = np.min(y_prim), np.max(y_prim)
                x_span = max(1e-6, x_max - x_min)
                y_span = max(1e-6, y_max - y_min)

                raw_dx = x_span * 0.15
                mag_x = 10 ** np.floor(np.log10(raw_dx))
                bar_dx = round(raw_dx / mag_x) * mag_x

                raw_dy = y_span * 0.25
                mag_y = 10 ** np.floor(np.log10(raw_dy))
                bar_dy = round(raw_dy / mag_y) * mag_y

                x0_bar = x_max - bar_dx
                y0_bar = y_min - 0.05 * y_span

                ax.plot([x0_bar, x_max], [y0_bar, y0_bar], color='black', lw=2.5)
                ax.plot([x_max, x_max], [y0_bar, y0_bar + bar_dy], color='black', lw=2.5)

                unit_short = "ms" if self.combo_time_unit.currentIndex() == 0 else ("s" if self.combo_time_unit.currentIndex() == 1 else "frames")
                ax.text(x0_bar + bar_dx / 2.0, y0_bar - 0.04 * y_span, f"{bar_dx:g} {unit_short}",
                        ha='center', va='top', fontsize=10, fontweight='bold', color='black')
                ax.text(x_max + 0.015 * x_span, y0_bar + bar_dy / 2.0, f"{bar_dy:g} {y_label}",
                        ha='left', va='center', fontsize=10, fontweight='bold', color='black')

        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 1 or (len(labels) == 1 and labels[0]):
            ax.legend(frameon=False, fontsize=9, loc='upper right')

        plt.tight_layout()
        dpi_val = dlg.spin_dpi.value()
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in ('.tif', '.tiff'):
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi_val, facecolor='white', edgecolor='none')
                buf.seek(0)
                img_arr = plt.imread(buf)
                img_uint8 = (img_arr[:, :, :3] * 255).astype(np.uint8)
                tifffile.imwrite(file_path, img_uint8, photometric='rgb')
            else:
                fig.savefig(file_path, format='jpeg', dpi=dpi_val, facecolor='white', edgecolor='none')
            plt.close(fig)
            QMessageBox.information(self, "Image Exported", f"Trace image successfully saved to:\n{file_path}")
        except Exception as e:
            plt.close(fig)
            QMessageBox.critical(self, "Export Error", f"Unable to export image:\n{str(e)}")

    # ==========================================
    # ADVANCED PEAK KINETICS UI
    # ==========================================
    def init_peak_analysis_ui(self):
        layout = QHBoxLayout(self.peak_analysis_widget)
        
        control_panel = QVBoxLayout()
        control_panel.setContentsMargins(10, 10, 10, 10)

        btn_back_main = QPushButton("← Return to Main Page")
        btn_back_main.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; padding: 8px;")
        btn_back_main.clicked.connect(lambda: self.central_stacked.setCurrentIndex(1))
        control_panel.addWidget(btn_back_main)

        title_label = QLabel("📊 Transient Analysis & Peak Kinetics")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        control_panel.addWidget(title_label)

        win_group = QGroupBox("Peak Alignment & Time Window")
        win_layout = QFormLayout()

        self.spin_pre_peak = NoWheelSpinBox()
        self.spin_pre_peak.setRange(1, 500)
        self.spin_pre_peak.setValue(18)
        self.spin_pre_peak.setMinimumHeight(26)
        self.spin_pre_peak.valueChanged.connect(self.update_peak_analysis_view)

        self.spin_total_window = NoWheelSpinBox()
        self.spin_total_window.setRange(5, 1000)
        self.spin_total_window.setValue(40)
        self.spin_total_window.setMinimumHeight(26)
        self.spin_total_window.valueChanged.connect(self.update_peak_analysis_view)

        self.combo_align_mode = NoWheelComboBox()
        self.combo_align_mode.addItems([
            "Max Slope (dF/dt)",
            "50% Rise Onset (Half-Rise)",
            "10% Rise Onset",
            "20% Rise Onset",
            "80% Rise Onset",
            "Peak Maximum (100%)"
        ])
        self.combo_align_mode.setCurrentIndex(0)
        self.combo_align_mode.setMinimumHeight(26)
        self.combo_align_mode.currentIndexChanged.connect(self.update_peak_analysis_view)

        win_layout.addRow("Pre-Peak Frames:", self.spin_pre_peak)
        win_layout.addRow("Total Window (Frames):", self.spin_total_window)
        win_layout.addRow("Temporal Alignment:", self.combo_align_mode)
        win_group.setLayout(win_layout)
        control_panel.addWidget(win_group)

        cursor_group = QGroupBox("Interactive Measurement Cursors & Visual Overlays")
        cursor_layout = QFormLayout()

        self.chk_use_cursors = QCheckBox("Enable Interactive Analysis Cursors")
        self.chk_use_cursors.setChecked(False)
        self.chk_use_cursors.toggled.connect(self.on_cursors_toggled)
        cursor_layout.addRow(self.chk_use_cursors)

        self.chk_show_rise_fit = QCheckBox("Show Linear Fit Rise Line (Rate of Rise)")
        self.chk_show_rise_fit.setChecked(True)
        self.chk_show_rise_fit.setStyleSheet("color: #76ff03; font-weight: bold;")
        self.chk_show_rise_fit.toggled.connect(self.update_peak_analysis_view)
        cursor_layout.addRow(self.chk_show_rise_fit)

        self.chk_show_fit_lines = QCheckBox("Show Fitted Decay Curve (Tau) & FWHM Line")
        self.chk_show_fit_lines.setChecked(True)
        self.chk_show_fit_lines.setStyleSheet("color: #0288d1; font-weight: bold;")
        self.chk_show_fit_lines.toggled.connect(self.update_peak_analysis_view)
        cursor_layout.addRow(self.chk_show_fit_lines)

        cursor_group.setLayout(cursor_layout)
        control_panel.addWidget(cursor_group)

        filter_group = QGroupBox("ROI & Peak Selection")
        filter_layout = QFormLayout()
        
        self.combo_peak_roi = NoWheelComboBox()
        self.combo_peak_roi.currentIndexChanged.connect(self.on_peak_roi_changed)
        filter_layout.addRow("Select ROI:", self.combo_peak_roi)

        self.combo_single_peak = NoWheelComboBox()
        self.combo_single_peak.addItem("All Peaks (Overlaid)")
        self.combo_single_peak.setMinimumHeight(26)
        self.combo_single_peak.currentIndexChanged.connect(self.on_single_peak_combo_changed)
        filter_layout.addRow("Select Peak:", self.combo_single_peak)

        self.chk_exclude_active_peak = QCheckBox("Exclude Active Peak from Ensemble Averaging & Metrics")
        self.chk_exclude_active_peak.setChecked(False)
        self.chk_exclude_active_peak.setEnabled(False)
        self.chk_exclude_active_peak.setStyleSheet("color: #ff5252; font-weight: bold;")
        self.chk_exclude_active_peak.toggled.connect(self.on_exclude_active_peak_toggled)
        filter_layout.addRow(self.chk_exclude_active_peak)

        self.radio_sem = QRadioButton("SEM (Standard Error)")
        self.radio_sem.setChecked(True)
        self.radio_sem.toggled.connect(self.update_peak_analysis_view)
        self.radio_sd = QRadioButton("SD (Standard Deviation)")
        self.radio_sd.toggled.connect(self.update_peak_analysis_view)

        err_layout = QHBoxLayout()
        err_layout.addWidget(self.radio_sem)
        err_layout.addWidget(self.radio_sd)
        filter_layout.addRow("Error Shading Mode:", err_layout)
        
        filter_group.setLayout(filter_layout)
        control_panel.addWidget(filter_group)

        btn_export_layout = QVBoxLayout()
        
        self.btn_export_pdf_report = QPushButton("📄 Generate Final Summary PDF Report")
        self.btn_export_pdf_report.setMinimumHeight(34)
        self.btn_export_pdf_report.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 13px;")
        self.btn_export_pdf_report.clicked.connect(self.generate_pdf_report)
        btn_export_layout.addWidget(self.btn_export_pdf_report)

        self.btn_export_peaks = QPushButton("💾 Export Peak Kinetics Table (Excel)")
        self.btn_export_peaks.setMinimumHeight(28)
        self.btn_export_peaks.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_export_peaks.clicked.connect(self.export_peaks_table)
        
        self.btn_export_mean_trace = QPushButton("📈 Export Mean Peak Trace (Mean ± Error)")
        self.btn_export_mean_trace.setMinimumHeight(28)
        self.btn_export_mean_trace.setStyleSheet("background-color: #0288d1; color: white; font-weight: bold;")
        self.btn_export_mean_trace.clicked.connect(self.export_mean_peak_trace)

        btn_export_layout.addWidget(self.btn_export_peaks)
        btn_export_layout.addWidget(self.btn_export_mean_trace)
        control_panel.addLayout(btn_export_layout)

        self.table_metrics = QTableWidget()
        self.table_metrics.setColumnCount(12)
        self.table_metrics.setHorizontalHeaderLabels([
            "Enabled", "ROI", "Peak #", "Peak Amp", "Rise (ms)", "Rate Rise (Fit)", "FWHM (ms)", "Tau (ms)", "Decay T50% (ms)", "Decay T90% (ms)", "AUC (ms)", "IEI (s)"
        ])
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_metrics.itemChanged.connect(self.on_table_item_changed)
        control_panel.addWidget(self.table_metrics)

        layout.addLayout(control_panel, stretch=1)

        right_layout = QVBoxLayout()
        
        self.peak_plot_widget = pg.PlotWidget(title="Overlay of Individual Transients (Drag Lines to Trim Analysis)")
        self.peak_plot_widget.setLabel('bottom', 'Relative Time')
        self.peak_plot_widget.setLabel('left', 'ΔF / F0')
        self.peak_plot_widget.showGrid(x=True, y=True)
        right_layout.addWidget(self.peak_plot_widget, stretch=1)

        self.mean_peak_plot_widget = pg.PlotWidget(title="Ensemble Mean Peak Trace (Mean Curve + Error Shading)")
        self.mean_peak_plot_widget.setLabel('bottom', 'Relative Time')
        self.mean_peak_plot_widget.setLabel('left', 'ΔF / F0')
        self.mean_peak_plot_widget.showGrid(x=True, y=True)
        right_layout.addWidget(self.mean_peak_plot_widget, stretch=1)

        layout.addLayout(right_layout, stretch=2)

    def on_cursors_toggled(self, checked):
        if not checked:
            self.remove_cursors_from_plot()
        self.update_peak_analysis_view()

    def remove_cursors_from_plot(self):
        if self.v_line_start:
            self.peak_plot_widget.removeItem(self.v_line_start)
            self.v_line_start = None
        if self.v_line_end:
            self.peak_plot_widget.removeItem(self.v_line_end)
            self.v_line_end = None
        if self.h_line_base:
            self.peak_plot_widget.removeItem(self.h_line_base)
            self.h_line_base = None

    def on_cursor_moved(self):
        selected_roi_idx = self.combo_peak_roi.currentIndex()
        selected_peak_idx = self.combo_single_peak.currentIndex()

        if selected_roi_idx > 0 and selected_peak_idx > 0:
            roi_idx = selected_roi_idx - 1
            peak_num = selected_peak_idx
            key = (roi_idx, peak_num)

            self.peak_cursors_map[key] = {
                'start': self.v_line_start.value() if self.v_line_start else -1.0,
                'end': self.v_line_end.value() if self.v_line_end else 1.0,
                'base': self.h_line_base.value() if self.h_line_base else 0.0
            }

        self.update_peak_analysis_view(rebuild_table=True, create_cursors=False)

    def open_peak_analysis_page(self):
        self.update_peak_analysis_roi_combo()
        self.update_single_peak_combo()
        self.update_peak_analysis_view()
        self.central_stacked.setCurrentIndex(2)

    def update_peak_analysis_roi_combo(self):
        self.combo_peak_roi.blockSignals(True)
        self.combo_peak_roi.clear()
        self.combo_peak_roi.addItem("All ROIs (Overlaid Means)")
        for i, roi in enumerate(self.rois):
            self.combo_peak_roi.addItem(f"ROI {i+1}")
        self.combo_peak_roi.blockSignals(False)

    def on_peak_roi_changed(self):
        self.update_single_peak_combo()
        self.update_peak_analysis_view()

    def update_single_peak_combo(self):
        self.combo_single_peak.blockSignals(True)
        self.combo_single_peak.clear()
        self.combo_single_peak.addItem("All Peaks (Overlaid)")
        
        selected_roi_idx = self.combo_peak_roi.currentIndex()
        data = self.calculate_data()

        if data and selected_roi_idx > 0:
            roi_idx = selected_roi_idx - 1
            if roi_idx < len(data['rois_data']):
                peaks = data['rois_data'][roi_idx]['peaks_indices']
                for p_num in range(len(peaks)):
                    self.combo_single_peak.addItem(f"Peak {p_num + 1}")

        self.combo_single_peak.setCurrentIndex(0)
        self.combo_single_peak.blockSignals(False)
        self.update_exclude_checkbox_state()

    def on_single_peak_combo_changed(self):
        self.update_exclude_checkbox_state()
        self.update_peak_analysis_view()

    def update_exclude_checkbox_state(self):
        selected_roi_idx = self.combo_peak_roi.currentIndex()
        selected_peak_idx = self.combo_single_peak.currentIndex()

        if selected_roi_idx > 0 and selected_peak_idx > 0:
            roi_idx = selected_roi_idx - 1
            peak_num = selected_peak_idx
            is_disabled = (roi_idx, peak_num) in self.disabled_peaks

            self.chk_exclude_active_peak.blockSignals(True)
            self.chk_exclude_active_peak.setChecked(is_disabled)
            self.chk_exclude_active_peak.setEnabled(True)
            self.chk_exclude_active_peak.blockSignals(False)
        else:
            self.chk_exclude_active_peak.blockSignals(True)
            self.chk_exclude_active_peak.setChecked(False)
            self.chk_exclude_active_peak.setEnabled(False)
            self.chk_exclude_active_peak.blockSignals(False)

    def on_exclude_active_peak_toggled(self, checked):
        selected_roi_idx = self.combo_peak_roi.currentIndex()
        selected_peak_idx = self.combo_single_peak.currentIndex()

        if selected_roi_idx > 0 and selected_peak_idx > 0:
            roi_idx = selected_roi_idx - 1
            peak_num = selected_peak_idx

            if checked:
                self.disabled_peaks.add((roi_idx, peak_num))
            else:
                self.disabled_peaks.discard((roi_idx, peak_num))

            self.update_peak_analysis_view(rebuild_table=True)

    def get_time_axis(self, num_points):
        dt_ms = self.spin_frame_interval_ms.value()
        unit_idx = self.combo_time_unit.currentIndex()
        
        frames = np.arange(num_points)
        if unit_idx == 0:
            return frames * dt_ms, "Time (ms)"
        elif unit_idx == 1:
            return frames * (dt_ms / 1000.0), "Time (s)"
        else:
            return frames, "Frames"

    def on_table_item_changed(self, item):
        if item.column() == 0:
            roi_name = self.table_metrics.item(item.row(), 1).text()
            roi_idx = int(roi_name.replace("ROI ", "")) - 1
            p_num = int(self.table_metrics.item(item.row(), 2).text())
            
            key = (roi_idx, p_num)
            if item.checkState() == Qt.CheckState.Unchecked:
                self.disabled_peaks.add(key)
            else:
                self.disabled_peaks.discard(key)
                
            self.update_exclude_checkbox_state()
            self.update_peak_analysis_view(rebuild_table=False)

    def fit_decay_tau(self, t_arr, y_arr):
        def mono_exp(t, a, tau, c):
            return a * np.exp(-t / tau) + c

        try:
            p0 = (y_arr[0] - y_arr[-1], max(0.1, (t_arr[-1] - t_arr[0]) / 3), y_arr[-1])
            bounds = ([0, 1e-4, -10], [10, 100000, 10])
            popt, _ = curve_fit(mono_exp, t_arr, y_arr, p0=p0, bounds=bounds, maxfev=1500)
            return popt
        except Exception:
            return None

    def update_peak_analysis_view(self, rebuild_table=True, create_cursors=True):
        data = self.calculate_data()
        if not data:
            return

        self.peak_plot_widget.clear()
        self.mean_peak_plot_widget.clear()

        self.v_line_start = None
        self.v_line_end = None
        self.h_line_base = None

        if rebuild_table:
            self.table_metrics.blockSignals(True)
            self.table_metrics.setRowCount(0)

        pre_frames = self.spin_pre_peak.value()
        total_frames = self.spin_total_window.value()
        selected_roi_idx = self.combo_peak_roi.currentIndex()
        selected_peak_idx = self.combo_single_peak.currentIndex()
        align_mode = self.combo_align_mode.currentText()
        dt_ms = self.spin_frame_interval_ms.value()
        dt_sec = dt_ms / 1000.0

        row_count = 0

        unit_idx = self.combo_time_unit.currentIndex()
        time_unit_label = "ms" if unit_idx == 0 else ("s" if unit_idx == 1 else "Frames")
        
        self.peak_plot_widget.setLabel('bottom', f'Time ({align_mode}) [{time_unit_label}]')
        self.mean_peak_plot_widget.setLabel('bottom', f'Time ({align_mode}) [{time_unit_label}]')

        all_valid_traces = []

        active_cursor_start = -dt_ms * 5
        active_cursor_end = dt_ms * 10
        active_cursor_base = 0.0

        for i, r in enumerate(data['rois_data']):
            if selected_roi_idx > 0 and (selected_roi_idx - 1) != i:
                continue

            trace = r['df_f0']
            peaks = r['peaks_indices']
            color = self._get_roi_display_color(i)

            for p_num, p_idx in enumerate(peaks):
                peak_id = p_num + 1
                
                if selected_roi_idx > 0 and selected_peak_idx > 0 and selected_peak_idx != peak_id:
                    continue

                is_enabled = (i, peak_id) not in self.disabled_peaks

                start_f = max(0, p_idx - pre_frames)
                end_f = min(len(trace), start_f + total_frames)
                sub_trace = np.copy(trace[start_f:end_f])

                peak_val = trace[p_idx]
                min_pre = np.min(sub_trace[:(p_idx - start_f)]) if (p_idx - start_f) > 0 else sub_trace[0]
                amp = peak_val - min_pre
                rising_segment = trace[start_f:p_idx]

                if "Max Slope" in align_mode:
                    if len(rising_segment) > 1:
                        dF = np.diff(rising_segment)
                        t_ref_idx = start_f + np.argmax(dF)
                    else:
                        t_ref_idx = p_idx
                elif "Peak Maximum" in align_mode:
                    t_ref_idx = p_idx
                else:
                    pct_val = float(align_mode.split("%")[0]) / 100.0
                    target_val = min_pre + pct_val * amp
                    idx_pct = np.where(rising_segment >= target_val)[0]
                    t_ref_idx = (start_f + idx_pct[0]) if len(idx_pct) > 0 else start_f

                rel_frames = np.arange(len(sub_trace)) - (t_ref_idx - start_f)
                if unit_idx == 0:
                    x_axis = rel_frames * dt_ms
                elif unit_idx == 1:
                    x_axis = rel_frames * dt_sec
                else:
                    x_axis = rel_frames

                pen_style = Qt.PenStyle.SolidLine if is_enabled else Qt.PenStyle.DotLine
                alpha_val = 0.9 if is_enabled else 0.25
                pen_w = 2.5 if (selected_peak_idx > 0 and is_enabled) else 1.5
                self.peak_plot_widget.plot(x_axis, sub_trace, pen=pg.mkPen(color, width=pen_w, style=pen_style, alpha=alpha_val))

                if is_enabled and len(sub_trace) == total_frames:
                    all_valid_traces.append(sub_trace)

                key = (i, peak_id)
                if key in self.peak_cursors_map:
                    c_info = self.peak_cursors_map[key]
                    c_start, c_end, c_base = c_info['start'], c_info['end'], c_info['base']
                else:
                    c_start = float(x_axis[0]) if len(x_axis) > 0 else -dt_ms
                    c_end = float(x_axis[-1]) if len(x_axis) > 0 else dt_ms * 10
                    c_base = float(min_pre)
                    self.peak_cursors_map[key] = {'start': c_start, 'end': c_end, 'base': c_base}

                active_cursor_start = c_start
                active_cursor_end = c_end
                active_cursor_base = c_base

                if self.chk_use_cursors.isChecked():
                    mask = (x_axis >= c_start) & (x_axis <= c_end)
                    if not np.any(mask):
                        mask = np.ones(len(x_axis), dtype=bool)
                    x_calc = x_axis[mask]
                    y_calc = sub_trace[mask] - c_base
                else:
                    x_calc = x_axis
                    y_calc = sub_trace - min_pre

                peak_amp_val = np.max(y_calc) if len(y_calc) > 0 else amp
                x_calc_ms = rel_frames[mask if self.chk_use_cursors.isChecked() else np.ones(len(x_axis), dtype=bool)] * dt_ms

                if len(y_calc) > 1 and peak_amp_val > 0:
                    peak_loc = np.argmax(y_calc)
                    y_rise = y_calc[:peak_loc + 1]
                    x_rise_ms = x_calc_ms[:peak_loc + 1]
                    x_rise_plot = x_calc[:peak_loc + 1]

                    t_0_interp_ms = np.interp(0.0, y_rise, x_rise_ms)
                    t_100_interp_ms = x_rise_ms[peak_loc]

                    rise_time_ms = max(0.1, t_100_interp_ms - t_0_interp_ms)
                else:
                    peak_loc = 0
                    y_rise = y_calc
                    x_rise_ms = x_calc_ms
                    x_rise_plot = x_calc
                    rise_time_ms = dt_ms

                if len(y_rise) >= 2:
                    slope_m, intercept_q = np.polyfit(x_rise_ms, y_rise, 1)
                    rate_of_rise_fit = max(0.0, slope_m)
                else:
                    rate_of_rise_fit = peak_amp_val / rise_time_ms if rise_time_ms > 0 else 0
                    slope_m, intercept_q = 0.0, 0.0

                if self.chk_show_rise_fit.isChecked() and len(y_rise) >= 2:
                    y_rise_fit_line = slope_m * x_rise_ms + intercept_q
                    if self.chk_use_cursors.isChecked():
                        y_rise_fit_line += c_base
                    else:
                        y_rise_fit_line += min_pre

                    rise_fit_curve = pg.PlotCurveItem(
                        x_rise_plot, y_rise_fit_line,
                        pen=pg.mkPen('#76ff03', width=2.5, style=Qt.PenStyle.DashLine)
                    )
                    self.peak_plot_widget.addItem(rise_fit_curve)

                half_max_val = 0.5 * peak_amp_val
                idx_fwhm = np.where(y_calc >= half_max_val)[0]
                fwhm_ms = (idx_fwhm[-1] - idx_fwhm[0]) * dt_ms if len(idx_fwhm) > 1 else 0

                if self.chk_show_fit_lines.isChecked() and len(idx_fwhm) > 1:
                    x_fwhm_start = x_calc[idx_fwhm[0]]
                    x_fwhm_end = x_calc[idx_fwhm[-1]]
                    y_fwhm_level = (half_max_val + c_base) if self.chk_use_cursors.isChecked() else (half_max_val + min_pre)
                    
                    fwhm_line = pg.PlotCurveItem(
                        [x_fwhm_start, x_fwhm_end], [y_fwhm_level, y_fwhm_level],
                        pen=pg.mkPen('#00e5ff', width=3, style=Qt.PenStyle.SolidLine)
                    )
                    self.peak_plot_widget.addItem(fwhm_line)

                falling_segment = y_calc[peak_loc:]
                x_falling_ms = np.arange(len(falling_segment)) * dt_ms
                x_falling_plot = x_calc[peak_loc:] if len(x_calc) > peak_loc else x_falling_ms

                fit_popt = self.fit_decay_tau(x_falling_ms, falling_segment) if len(falling_segment) > 2 else None
                tau_ms = fit_popt[1] if fit_popt is not None else np.nan

                if self.chk_show_fit_lines.isChecked() and fit_popt is not None and len(x_falling_plot) > 2:
                    t_fit_dense_ms = np.linspace(x_falling_ms[0], x_falling_ms[-1], 100)
                    t_fit_dense_plot = np.linspace(x_falling_plot[0], x_falling_plot[-1], 100)
                    
                    y_fit_dense = fit_popt[0] * np.exp(-t_fit_dense_ms / fit_popt[1]) + fit_popt[2]
                    if self.chk_use_cursors.isChecked():
                        y_fit_dense += c_base
                    else:
                        y_fit_dense += min_pre

                    tau_curve = pg.PlotCurveItem(
                        t_fit_dense_plot, y_fit_dense,
                        pen=pg.mkPen('#ff007f', width=2.5, style=Qt.PenStyle.DashLine)
                    )
                    self.peak_plot_widget.addItem(tau_curve)

                target_decay_50 = 0.5 * peak_amp_val
                decay_t50_ms = np.interp(-target_decay_50, -falling_segment, x_falling_ms) if len(falling_segment) > 1 else 0.0

                target_decay_90 = 0.1 * peak_amp_val
                decay_t90_ms = np.interp(-target_decay_90, -falling_segment, x_falling_ms) if len(falling_segment) > 1 else 0.0

                if hasattr(np, 'trapezoid'):
                    auc_ms = np.trapezoid(np.maximum(0, y_calc), dx=dt_ms)
                else:
                    auc_ms = np.trapz(np.maximum(0, y_calc), dx=dt_ms)

                iei_sec = (peaks[p_num] - peaks[p_num - 1]) * dt_sec if p_num > 0 else np.nan

                if rebuild_table:
                    self.table_metrics.insertRow(row_count)
                    chk_item = QTableWidgetItem()
                    chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                    chk_item.setCheckState(Qt.CheckState.Checked if is_enabled else Qt.CheckState.Unchecked)
                    
                    self.table_metrics.setItem(row_count, 0, chk_item)
                    self.table_metrics.setItem(row_count, 1, QTableWidgetItem(r['name']))
                    self.table_metrics.setItem(row_count, 2, QTableWidgetItem(str(peak_id)))
                    self.table_metrics.setItem(row_count, 3, QTableWidgetItem(f"{peak_amp_val:.4f}"))
                    self.table_metrics.setItem(row_count, 4, QTableWidgetItem(f"{rise_time_ms:.2f}"))
                    self.table_metrics.setItem(row_count, 5, QTableWidgetItem(f"{rate_of_rise_fit:.6f}"))
                    self.table_metrics.setItem(row_count, 6, QTableWidgetItem(f"{fwhm_ms:.2f}"))
                    self.table_metrics.setItem(row_count, 7, QTableWidgetItem(f"{tau_ms:.2f}" if not np.isnan(tau_ms) else "N/A"))
                    self.table_metrics.setItem(row_count, 8, QTableWidgetItem(f"{decay_t50_ms:.2f}"))
                    self.table_metrics.setItem(row_count, 9, QTableWidgetItem(f"{decay_t90_ms:.2f}"))
                    self.table_metrics.setItem(row_count, 10, QTableWidgetItem(f"{auc_ms:.2f}"))
                    self.table_metrics.setItem(row_count, 11, QTableWidgetItem(f"{iei_sec:.2f}" if not np.isnan(iei_sec) else "N/A"))
                    row_count += 1

        if self.chk_use_cursors.isChecked():
            self.v_line_start = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#00e676', width=2, style=Qt.PenStyle.DashLine), pos=active_cursor_start)
            self.v_line_end = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ff1744', width=2, style=Qt.PenStyle.DashLine), pos=active_cursor_end)
            self.h_line_base = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#ffea00', width=2, style=Qt.PenStyle.SolidLine), pos=active_cursor_base)

            self.v_line_start.sigPositionChangeFinished.connect(self.on_cursor_moved)
            self.v_line_end.sigPositionChangeFinished.connect(self.on_cursor_moved)
            self.h_line_base.sigPositionChangeFinished.connect(self.on_cursor_moved)

            self.peak_plot_widget.addItem(self.v_line_start)
            self.peak_plot_widget.addItem(self.v_line_end)
            self.peak_plot_widget.addItem(self.h_line_base)

        if rebuild_table:
            self.table_metrics.blockSignals(False)

        if all_valid_traces:
            traces_matrix = np.array(all_valid_traces)
            mean_trace = np.mean(traces_matrix, axis=0)
            
            if self.radio_sem.isChecked():
                err_trace = np.std(traces_matrix, axis=0) / np.sqrt(len(traces_matrix))
            else:
                err_trace = np.std(traces_matrix, axis=0)

            rel_frames = np.arange(total_frames) - pre_frames
            if unit_idx == 0:
                x_mean = rel_frames * dt_ms
            elif unit_idx == 1:
                x_mean = rel_frames * dt_sec
            else:
                x_mean = rel_frames

            curve_upper = mean_trace + err_trace
            curve_lower = mean_trace - err_trace

            fill = pg.FillBetweenItem(
                pg.PlotCurveItem(x_mean, curve_upper),
                pg.PlotCurveItem(x_mean, curve_lower),
                brush=pg.mkBrush(100, 181, 246, 100)
            )
            self.mean_peak_plot_widget.addItem(fill)
            self.mean_peak_plot_widget.plot(x_mean, mean_trace, pen=pg.mkPen('#1565c0', width=3))

    def generate_pdf_report(self):
        data = self.calculate_data()
        if not data:
            QMessageBox.warning(self, "Warning", "No active analysis data available to build the PDF report.")
            return

        default_pdf_path = os.path.join(self.last_dir, "Calcium_Analysis_Summary_Report.pdf") if self.last_dir else "Calcium_Analysis_Summary_Report.pdf"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Summary Report", default_pdf_path, "PDF Document (*.pdf)")
        if not file_path:
            return

        self.last_dir = os.path.dirname(file_path)

        num_frames = len(self.stack)
        w_frames = self.spin_coincidence_window.value()
        bin_matrix = []
        
        for r in data['rois_data']:
            p_indices = r['peaks_indices']
            if len(p_indices) > 0:
                vec = np.zeros(num_frames, dtype=float)
                for p in p_indices:
                    vec[max(0, p - w_frames):min(num_frames, p + w_frames + 1)] = 1.0
                bin_matrix.append(vec)

        num_active_nodes = len(bin_matrix)
        if num_active_nodes >= 2:
            bin_matrix = np.array(bin_matrix)
            event_corr = np.zeros((num_active_nodes, num_active_nodes))
            for i in range(num_active_nodes):
                for j in range(num_active_nodes):
                    if i == j:
                        event_corr[i, j] = 1.0
                    else:
                        inter = np.sum((bin_matrix[i] == 1.0) & (bin_matrix[j] == 1.0))
                        union = np.sum((bin_matrix[i] == 1.0) | (bin_matrix[j] == 1.0))
                        event_corr[i, j] = inter / union if union > 0 else 0.0
            
            mask_offdiag = ~np.eye(num_active_nodes, dtype=bool)
            mean_jaccard = np.mean(event_corr[mask_offdiag])
        else:
            mean_jaccard = 0.0
            event_corr = np.ones((1, 1))

        num_rois_total = len(data['rois_data'])
        if num_rois_total >= 2:
            df_dict = {r['name']: r['df_f0'] for r in data['rois_data']}
            df_matrix = pd.DataFrame(df_dict)
            c_matrix = df_matrix.corr().values
            mask_c = ~np.eye(num_rois_total, dtype=bool)
            mean_pearson = np.nanmean(c_matrix[mask_c])

            ecr_mat = np.zeros((num_active_nodes, num_active_nodes))
            peaks_list = [r['peaks_indices'] for r in data['rois_data'] if len(r['peaks_indices']) > 0]
            for i in range(num_active_nodes):
                for j in range(num_active_nodes):
                    if i == j:
                        ecr_mat[i, j] = 1.0
                    else:
                        pA, pB = peaks_list[i], peaks_list[j]
                        cA = sum(1 for pa in pA if np.any(np.abs(pB - pa) <= w_frames))
                        cB = sum(1 for pb in pB if np.any(np.abs(pA - pb) <= w_frames))
                        denom = min(len(pA), len(pB))
                        ecr_mat[i, j] = min(cA, cB) / denom if denom > 0 else 0.0
            mean_ecr = np.mean(ecr_mat[mask_offdiag]) if num_active_nodes >= 2 else 0.0
        else:
            mean_pearson = 0.0
            mean_ecr = 0.0

        fig_hm, ax_hm = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(event_corr, cmap=self.combo_cmap.currentText(), vmin=0, vmax=1, ax=ax_hm, cbar=True,
                    xticklabels=[f"R{i+1}" for i in range(num_active_nodes)],
                    yticklabels=[f"R{i+1}" for i in range(num_active_nodes)])
        ax_hm.set_title("Discrete Peak Coincidence Matrix (Jaccard)", fontsize=10, fontweight='bold')
        plt.tight_layout()

        buf_hm = io.BytesIO()
        plt.savefig(buf_hm, format='png', dpi=200)
        plt.close(fig_hm)
        buf_hm.seek(0)

        pre_frames = self.spin_pre_peak.value()
        total_frames = self.spin_total_window.value()
        dt_ms = self.spin_frame_interval_ms.value()

        all_valid_traces = []
        for i, r in enumerate(data['rois_data']):
            trace = r['df_f0']
            for p_num, p_idx in enumerate(r['peaks_indices']):
                if (i, p_num + 1) not in self.disabled_peaks:
                    start_f = max(0, p_idx - pre_frames)
                    end_f = min(len(trace), start_f + total_frames)
                    sub_tr = trace[start_f:end_f]
                    if len(sub_tr) == total_frames:
                        all_valid_traces.append(sub_tr - sub_tr[0])

        fig_tr, ax_tr = plt.subplots(figsize=(5.5, 4.5))
        if all_valid_traces:
            mat = np.array(all_valid_traces)
            mean_tr = np.mean(mat, axis=0)
            sem_tr = np.std(mat, axis=0) / np.sqrt(len(mat))
            t_rel_ms = (np.arange(total_frames) - pre_frames) * dt_ms

            ax_tr.plot(t_rel_ms, mean_tr, color='#1565c0', lw=2, label="Mean dF/F0")
            ax_tr.fill_between(t_rel_ms, mean_tr - sem_tr, mean_tr + sem_tr, color='#64b5f6', alpha=0.4, label="SEM")
            ax_tr.set_xlabel("Time (ms)", fontsize=9)
            ax_tr.set_ylabel("dF / F0", fontsize=9)
            ax_tr.set_title(f"Ensemble Mean Peak Transient (N = {len(mat)})", fontsize=10, fontweight='bold')
            ax_tr.grid(True, linestyle='--', alpha=0.5)
            ax_tr.legend(loc="upper right", fontsize=8)
        else:
            ax_tr.text(0.5, 0.5, "No Active Transients", ha='center', va='center')
        
        plt.tight_layout()
        buf_tr = io.BytesIO()
        plt.savefig(buf_tr, format='png', dpi=200)
        plt.close(fig_tr)
        buf_tr.seek(0)

        rows = self.table_metrics.rowCount()
        cols = self.table_metrics.columnCount()
        headers = [self.table_metrics.horizontalHeaderItem(c).text() for c in range(cols)]

        table_data = []
        for r in range(rows):
            row_dict = {}
            for c in range(cols):
                if c == 0:
                    item = self.table_metrics.item(r, c)
                    row_dict[headers[c]] = "YES" if item.checkState() == Qt.CheckState.Checked else "NO"
                else:
                    item = self.table_metrics.item(r, c)
                    val_str = item.text() if item else ""
                    if val_str not in ["N/A", ""]:
                        try:
                            row_dict[headers[c]] = float(val_str)
                        except ValueError:
                            row_dict[headers[c]] = val_str
                    else:
                        row_dict[headers[c]] = np.nan
            table_data.append(row_dict)

        df_pdf = pd.DataFrame(table_data)
        df_pdf_enabled = df_pdf[df_pdf["Enabled"] == "YES"] if not df_pdf.empty else pd.DataFrame()

        param_cols = ["Peak Amp", "Rise (ms)", "Rate Rise (Fit)", "FWHM (ms)", "Tau (ms)", "Decay T50% (ms)", "Decay T90% (ms)", "AUC (ms)", "IEI (s)"]
        summary_rows = [["Parameter", "Mean Value", "Std Dev (SD)", "N Events"]]

        for col in param_cols:
            if not df_pdf_enabled.empty and col in df_pdf_enabled.columns:
                vals = pd.to_numeric(df_pdf_enabled[col], errors='coerce').dropna()
                if len(vals) > 0:
                    summary_rows.append([col, f"{vals.mean():.4f}", f"{vals.std():.4f}", str(len(vals))])
                else:
                    summary_rows.append([col, "N/A", "N/A", "0"])
            else:
                summary_rows.append([col, "N/A", "N/A", "0"])

        doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=colors.HexColor('#0f172a'), alignment=1
        )
        subtitle_style = ParagraphStyle(
            'DocSubTitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor('#475569'), alignment=1
        )
        heading_style = ParagraphStyle(
            'SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#1e40af'), spaceBefore=12, spaceAfter=6
        )

        elements = []

        elements.append(Paragraph("Calcium Imaging Analysis Summary Report", title_style))
        elements.append(Paragraph(f"Active Dataset: <b>{os.path.basename(self.current_stack_path or 'N/A')}</b> | Lead Author: {LEAD_AUTHOR}", subtitle_style))
        if CO_AUTHORS:
            elements.append(Paragraph(f"Co-Authors: {', '.join(CO_AUTHORS)}", subtitle_style))
        elements.append(Spacer(1, 10))

        sync_box_data = [
            [Paragraph(f"<b>Correlation Method</b>", ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#065f46'))),
             Paragraph(f"<b>Mean Correlation Value</b>", ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#065f46'), alignment=2))],
            [Paragraph("Continuous Pearson Correlation (r)", ParagraphStyle('P1', fontName='Helvetica', fontSize=9)), Paragraph(f"{mean_pearson:.4f}", ParagraphStyle('P1_v', fontName='Helvetica-Bold', fontSize=9, alignment=2))],
            [Paragraph("Event Coincidence Rate (ECR)", ParagraphStyle('P2', fontName='Helvetica', fontSize=9)), Paragraph(f"{mean_ecr:.4f}", ParagraphStyle('P2_v', fontName='Helvetica-Bold', fontSize=9, alignment=2))],
            [Paragraph("Discrete Peak Coincidence Index (Jaccard)", ParagraphStyle('P3', fontName='Helvetica', fontSize=9)), Paragraph(f"{mean_jaccard:.4f}", ParagraphStyle('P3_v', fontName='Helvetica-Bold', fontSize=9, alignment=2))]
        ]
        sync_table = Table(sync_box_data, colWidths=[360, 180])
        sync_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#d1fae5')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#10b981')),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#10b981'))
        ]))
        elements.append(sync_table)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("1. Graphical Network & Transient Analysis", heading_style))
        img_hm = RLImage(buf_hm, width=260, height=210)
        img_tr = RLImage(buf_tr, width=260, height=210)
        
        graphs_table = Table([[img_hm, img_tr]], colWidths=[270, 270])
        graphs_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(graphs_table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("2. Overall Average Peak Kinetics Metrics (ms)", heading_style))
        param_table = Table(summary_rows, colWidths=[180, 120, 120, 120])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        elements.append(param_table)

        try:
            doc.build(elements)
            QMessageBox.information(self, "PDF Report Exported", f"Summary PDF report successfully generated:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Export Error", f"Unable to generate PDF report:\n{str(e)}")

    def export_mean_peak_trace(self):
        data = self.calculate_data()
        if not data:
            QMessageBox.warning(self, "Warning", "No data available to export.")
            return

        default_save_path = os.path.join(self.last_dir, "mean_peak_trace.xlsx") if self.last_dir else "mean_peak_trace.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Mean Peak Trace", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
        if not file_path:
            return

        self.last_dir = os.path.dirname(file_path)

        pre_frames = self.spin_pre_peak.value()
        total_frames = self.spin_total_window.value()
        selected_roi_idx = self.combo_peak_roi.currentIndex()
        dt_ms = self.spin_frame_interval_ms.value()

        rel_frames = np.arange(total_frames) - pre_frames
        time_rel_ms = rel_frames * dt_ms

        export_dict = {
            'Relative Frame': rel_frames,
            'Relative Time (ms)': time_rel_ms
        }

        rois_to_process = range(len(data['rois_data'])) if selected_roi_idx == 0 else [selected_roi_idx - 1]
        all_combined_traces = []

        for i in rois_to_process:
            r = data['rois_data'][i]
            trace = r['df_f0']
            peaks = r['peaks_indices']
            roi_traces = []

            for p_num, p_idx in enumerate(peaks):
                peak_id = p_num + 1
                if (i, peak_id) in self.disabled_peaks:
                    continue

                start_f = max(0, p_idx - pre_frames)
                end_f = min(len(trace), start_f + total_frames)
                sub_trace = np.copy(trace[start_f:end_f])

                if len(sub_trace) == total_frames:
                    roi_traces.append(sub_trace)
                    all_combined_traces.append(sub_trace)

            if roi_traces:
                mat = np.array(roi_traces)
                m_tr = np.mean(mat, axis=0)
                sd_tr = np.std(mat, axis=0)
                sem_tr = sd_tr / np.sqrt(len(roi_traces))

                export_dict[f"{r['name']}_Mean"] = m_tr
                export_dict[f"{r['name']}_SD"] = sd_tr
                export_dict[f"{r['name']}_SEM"] = sem_tr

        if selected_roi_idx == 0 and all_combined_traces:
            mat_all = np.array(all_combined_traces)
            m_all = np.mean(mat_all, axis=0)
            sd_all = np.std(mat_all, axis=0)
            sem_all = sd_all / np.sqrt(len(all_combined_traces))

            export_dict["ALL_ROIS_Global_Mean"] = m_all
            export_dict["ALL_ROIS_Global_SD"] = sd_all
            export_dict["ALL_ROIS_Global_SEM"] = sem_all

        df_export = pd.DataFrame(export_dict)

        try:
            if file_path.endswith('.xlsx'):
                df_export.to_excel(file_path, index=False)
            else:
                df_export.to_csv(file_path, index=False, decimal='.', sep=',')
            QMessageBox.information(self, "Export Complete", f"Ensemble mean peak trace successfully exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Unable to save file:\n{str(e)}")

    def export_peaks_table(self):
        if self.table_metrics.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No data in table to export.")
            return

        default_save_path = os.path.join(self.last_dir, "calcium_peaks_kinetics.xlsx") if self.last_dir else "calcium_peaks_kinetics.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Peak Kinetics", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
        if not file_path:
            return

        self.last_dir = os.path.dirname(file_path)

        rows = self.table_metrics.rowCount()
        cols = self.table_metrics.columnCount()
        headers = [self.table_metrics.horizontalHeaderItem(c).text() for c in range(cols)]

        table_data = []
        for r in range(rows):
            row_dict = {}
            for c in range(cols):
                if c == 0:
                    item = self.table_metrics.item(r, c)
                    row_dict[headers[c]] = "YES" if item.checkState() == Qt.CheckState.Checked else "NO"
                else:
                    item = self.table_metrics.item(r, c)
                    val_str = item.text() if item else ""
                    if val_str not in ["N/A", ""]:
                        try:
                            val_num = float(val_str)
                            row_dict[headers[c]] = val_num
                        except ValueError:
                            row_dict[headers[c]] = val_str
                    else:
                        row_dict[headers[c]] = np.nan
            table_data.append(row_dict)

        df_peaks = pd.DataFrame(table_data)

        df_enabled = df_peaks[df_peaks["Enabled"] == "YES"].copy()
        numeric_cols = ["Peak Amp", "Rise (ms)", "Rate Rise (Fit)", "FWHM (ms)", "Tau (ms)", "Decay T50% (ms)", "Decay T90% (ms)", "AUC (ms)", "IEI (s)"]
        for col in numeric_cols:
            if col in df_enabled.columns:
                df_enabled[col] = pd.to_numeric(df_enabled[col], errors='coerce')

        df_summary = df_enabled.groupby("ROI")[numeric_cols].agg(['mean', 'std']).reset_index()

        try:
            if file_path.endswith('.xlsx'):
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df_peaks.to_excel(writer, sheet_name='Individual Peaks', index=False)
                    df_summary.to_excel(writer, sheet_name='ROI Means and SD', index=True)
            else:
                df_peaks.to_csv(file_path, index=False, decimal='.', sep=',')

            QMessageBox.information(self, "Export Complete", f"Peak kinetics report successfully saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to save file:\n{str(e)}")

    def open_help_dialog(self):
        dialog = HelpGuideDialog(self)
        dialog.exec()

    # --- CLUSTER ENVELOPE DRAWING ---
    def draw_roi_cluster_envelope(self):
        if len(self.rois) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 ROIs are required to form a cluster envelope contour.")
            return

        roi_names = [f"ROI {i+1}" for i in range(len(self.rois))]
        dialog = ClusterSelectionDialog(roi_names, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_indices = dialog.get_selected_indices()
            if len(selected_indices) < 3:
                QMessageBox.warning(self, "Warning", "Please select at least 3 ROIs to form a closed contour boundary.")
                return

            points = []
            for idx in selected_indices:
                cx, cy = self._get_roi_center(self.rois[idx])
                points.append([cx, cy])

            points = np.array(points)
            
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])

                col = dialog.selected_color
                r, g, b, a = col.red(), col.green(), col.blue(), col.alpha()

                pen = pg.mkPen(color=(r, g, b, 255), width=3, style=Qt.PenStyle.DashLine)
                brush = pg.mkBrush(r, g, b, 60)

                curve_item = pg.PlotCurveItem(hull_points[:, 0], hull_points[:, 1], pen=pen, fillLevel=0, brush=brush)
                self.view_box.addItem(curve_item)

                centroid_x = np.mean(points[:, 0])
                centroid_y = np.mean(points[:, 1])

                text_item = pg.TextItem(f"Cluster ({len(selected_indices)} ROIs)", color=(r, g, b), anchor=(0.5, 0.5))
                text_item.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                text_item.setPos(centroid_x, centroid_y)
                self.view_box.addItem(text_item)

                self.overlay_history.append([curve_item, text_item])
                self.btn_undo_overlay.setEnabled(True)
                self.btn_clear_overlay.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Geometry Error", f"Unable to draw cluster contour:\n{str(e)}")

    def clear_all_overlays(self):
        for level in self.overlay_history:
            for item in level:
                self.view_box.removeItem(item)
        self.overlay_history.clear()
        self.btn_undo_overlay.setEnabled(False)
        self.btn_clear_overlay.setEnabled(False)

    def undo_last_overlay(self):
        if self.overlay_history:
            last_level = self.overlay_history.pop()
            for item in last_level:
                self.view_box.removeItem(item)
            
            if not self.overlay_history:
                self.btn_undo_overlay.setEnabled(False)
                self.btn_clear_overlay.setEnabled(False)

    def highlight_hub_neurons_peak_based(self):
        data = self.calculate_data()
        if data is None or len(data['rois_data']) < 2:
            QMessageBox.warning(self, "Warning", "At least 2 active ROIs are required to identify HUB neurons.")
            return

        start_f = max(0, self.spin_analysis_start.value() - 1)
        end_f = min(len(self.stack), self.spin_analysis_end.value())
        num_frames = end_f - start_f
        w_frames = self.spin_coincidence_window.value()

        bin_matrix = []
        active_rois_info = []

        for r_idx, r in enumerate(data['rois_data']):
            p_indices = r['peaks_indices']
            if len(p_indices) > 0:
                binary_vector = np.zeros(num_frames, dtype=float)
                for p in p_indices:
                    st = max(0, p - w_frames)
                    en = min(num_frames, p + w_frames + 1)
                    binary_vector[st:en] = 1.0

                bin_matrix.append(binary_vector)
                active_rois_info.append({'idx': r_idx, 'name': r['name']})

        num_active = len(active_rois_info)
        if num_active < 2:
            QMessageBox.warning(self, "Insufficient Active ROIs", "At least 2 ROIs with detected peaks are required to calculate peak coincidence HUBs.")
            return

        bin_matrix = np.array(bin_matrix)
        event_corr_matrix = np.zeros((num_active, num_active))

        for i in range(num_active):
            for j in range(num_active):
                if i == j:
                    event_corr_matrix[i, j] = 1.0
                else:
                    intersection = np.sum((bin_matrix[i] == 1.0) & (bin_matrix[j] == 1.0))
                    union = np.sum((bin_matrix[i] == 1.0) | (bin_matrix[j] == 1.0))
                    event_corr_matrix[i, j] = intersection / union if union > 0 else 0.0

        mean_coincidence = np.mean(event_corr_matrix, axis=1)
        sorted_indices = np.argsort(mean_coincidence)[::-1]

        num_hubs_to_show = min(3, num_active)
        hub_crowns = ["👑", "🥈", "🥉"]
        hub_colors = [(255, 215, 0), (192, 192, 192), (205, 127, 50)]

        current_level_items = []
        msg_text = "Top Peak-Coincidence HUB Neurons Identified:\n\n"

        for rank in range(num_hubs_to_show):
            local_idx = sorted_indices[rank]
            roi_original_idx = active_rois_info[local_idx]['idx']
            roi_name = active_rois_info[local_idx]['name']
            score = mean_coincidence[local_idx]

            center_x, center_y = self._get_roi_center(self.rois[roi_original_idx])

            crown = hub_crowns[rank]
            col = hub_colors[rank]

            hub_circle = pg.CircleROI([center_x - (25 + rank * 3), center_y - (25 + rank * 3)], [50 + rank * 6, 50 + rank * 6], pen=pg.mkPen(color=col, width=4 - rank, style=Qt.PenStyle.SolidLine))
            hub_circle.removeHandle(0)
            self.view_box.addItem(hub_circle)
            current_level_items.append(hub_circle)

            hub_label = pg.TextItem(f"{crown} {roi_name} ({score:.2f})", color=col, anchor=(0.5, 1.2))
            hub_label.setFont(QFont("Arial", 11 - rank, QFont.Weight.Bold))
            hub_label.setPos(center_x, center_y - (25 + rank * 3))
            self.view_box.addItem(hub_label)
            current_level_items.append(hub_label)

            msg_text += f"• **Rank {rank+1} ({crown}):** {roi_name} - Mean Jaccard Coincidence = **{score:.3f}**\n"

        self.overlay_history.append(current_level_items)
        self.btn_undo_overlay.setEnabled(True)
        self.btn_clear_overlay.setEnabled(True)

        QMessageBox.information(self, "Top HUB Neurons Identified", msg_text)

    def on_plot_clicked(self, event):
        if self.stack is None or not self.rois:
            return

        filter_idx = self.combo_roi_filter.currentIndex()
        if filter_idx == 0:
            return

        roi_idx = filter_idx - 1
        data = self.calculate_data()
        if not data or roi_idx >= len(data['rois_data']):
            return

        pos = event.scenePos()
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        
        x_val = mouse_point.x()
        dt_ms = self.spin_frame_interval_ms.value()
        unit_idx = self.combo_time_unit.currentIndex()

        if unit_idx == 0:
            rel_frame = int(round(x_val / dt_ms))
        elif unit_idx == 1:
            rel_frame = int(round(x_val / (dt_ms / 1000.0)))
        else:
            rel_frame = int(round(x_val))

        roi_info = data['rois_data'][roi_idx]
        current_peaks = [int(p) for p in roi_info['peaks_indices']]

        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            if 0 <= rel_frame < len(roi_info['df_f0']) and rel_frame not in current_peaks:
                current_peaks.append(int(rel_frame))
                current_peaks.sort()
                self.manual_peaks_override[roi_idx] = current_peaks
                self.update_all_traces()

        elif event.button() == Qt.MouseButton.RightButton and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            for p in current_peaks:
                if abs(p - rel_frame) <= 6:
                    current_peaks.remove(p)
                    self.manual_peaks_override[roi_idx] = current_peaks
                    self.update_all_traces()
                    break

    def load_file(self, file_path=None):
        if not file_path:
            file_filter = "Compatible Files (*.tif *.tiff *.nd2 *.avi *.mp4);;Image Stacks (*.tif *.tiff *.nd2);;Video Files (*.avi *.mp4);;All Files (*)"
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Primary Image Stack or Video (Channel 1)", self.last_dir, file_filter)
        
        if not file_path or not os.path.exists(file_path):
            return False

        loaded_stack = self._read_stack_from_path(file_path)
        if loaded_stack is not None and loaded_stack.ndim == 3:
            self.stack = loaded_stack
            self.current_stack_path = file_path
            self.last_dir = os.path.dirname(file_path)

            self.stack_ch2 = None
            self.stack_ch3 = None
            self.stack_ch2_path = None
            self.stack_ch3_path = None
            self.chk_overlay_multichannel.setEnabled(False)
            self.btn_clear_extra_ch.setEnabled(False)

            self.update_active_filename_banner()

            num_frames = len(self.stack)
            self.slider_frame.setMinimum(0)
            self.slider_frame.setMaximum(num_frames - 1)
            self.slider_frame.setValue(0)
            self.slider_frame.setEnabled(True)
            self.btn_show_max_proj.setEnabled(True)
            self.lbl_frame_info.setText(f"Frame: Max Projection (1..{num_frames})")

            self.combo_display_channel.setCurrentIndex(0)
            self.combo_display_channel.setEnabled(True)
            self.refresh_displayed_image(frame_idx=None)

            self.spin_analysis_start.setMaximum(num_frames)
            self.spin_analysis_end.setMaximum(num_frames)
            self.spin_analysis_end.setValue(num_frames)

            self.spin_f0_end.setMaximum(num_frames)
            self.spin_f0_start.setMaximum(num_frames)

            self.btn_load_ch2.setEnabled(True)
            self.btn_load_ch3.setEnabled(True)
            self.btn_set_bg.setEnabled(True)
            self.btn_add_roi.setEnabled(True)
            self.btn_freehand_roi.setEnabled(True)
            self.combo_bleach_mode.setEnabled(True)
            self.btn_save_session.setEnabled(True)
            self.btn_close_session.setEnabled(True)

            if self.bg_roi is None:
                self.set_bg_roi()
            return True
        return False

    def close_session(self):
        reply = QMessageBox.question(
            self, "Close Session", 
            "Are you sure you want to close the current analysis session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.clear_all_rois()
            self.clear_all_overlays()

            if self.bg_roi is not None:
                self.view_box.removeItem(self.bg_roi)
                self.bg_roi = None
            if self.bg_label is not None:
                self.view_box.removeItem(self.bg_label)
                self.bg_label = None

            self.stack = None
            self.stack_ch2 = None
            self.stack_ch3 = None
            self.current_stack_path = None
            self.stack_ch2_path = None
            self.stack_ch3_path = None
            self.roi_custom_colors.clear()
            self.update_active_filename_banner()
            self.manual_peaks_override.clear()
            self.disabled_peaks.clear()
            self.img_item.clear()
            self.plot_widget.clear()

            self.lbl_frame_info.setText("Frame: 0 / 0")
            self.slider_frame.setValue(0)
            self.slider_frame.setEnabled(False)
            self.btn_show_max_proj.setEnabled(False)
            self.combo_display_channel.setEnabled(False)

            self.btn_freehand_roi.setChecked(False)
            self.toggle_freehand_roi_mode(False)

            self.combo_bleach_mode.setCurrentIndex(0)
            self.combo_bleach_mode.setEnabled(False)

            self.btn_load_ch2.setEnabled(False)
            self.btn_load_ch3.setEnabled(False)
            self.btn_clear_extra_ch.setEnabled(False)
            self.chk_overlay_multichannel.setEnabled(False)
            self.btn_change_trace_color.setEnabled(False)
            self.btn_export_trace_img.setEnabled(False)
            self.btn_set_bg.setEnabled(False)
            self.btn_add_roi.setEnabled(False)
            self.btn_freehand_roi.setEnabled(False)
            self.btn_delete_selected_roi.setEnabled(False)
            self.btn_sync.setEnabled(False)
            self.btn_ecr_sync.setEnabled(False)
            self.btn_event_sync.setEnabled(False)
            self.btn_latency.setEnabled(False)
            self.btn_show_hub.setEnabled(False)
            self.btn_draw_cluster.setEnabled(False)
            self.btn_main_pdf_report.setEnabled(False)
            self.btn_go_peak_analysis.setEnabled(False)
            self.btn_export.setEnabled(False)
            self.btn_save_session.setEnabled(False)
            self.btn_close_session.setEnabled(False)

            QMessageBox.information(self, "Session Closed", "Analysis session closed successfully.")

    def on_frame_slider_changed(self, frame_idx):
        if self.stack is not None and 0 <= frame_idx < len(self.stack):
            self.lbl_frame_info.setText(f"Frame: {frame_idx + 1} / {len(self.stack)}")
            self.refresh_displayed_image(frame_idx=frame_idx)

    def set_bg_roi(self, pos=None, size=None):
        if self.stack is None:
            return
        if self.bg_roi is not None:
            self.view_box.removeItem(self.bg_roi)
            if self.bg_label is not None:
                self.view_box.removeItem(self.bg_label)

        pos = pos if pos else [10, 10]
        size = size if size else [40, 40]

        self.bg_roi = pg.RectROI(pos, size, pen=pg.mkPen('w', style=pg.QtCore.Qt.PenStyle.DashLine, width=2))
        self.bg_roi.addScaleHandle([1, 1], [0, 0])
        self.view_box.addItem(self.bg_roi)

        self.bg_label = pg.TextItem("BG", color=(200, 200, 200), anchor=(0.5, 0.5))
        self.bg_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.view_box.addItem(self.bg_label)

        self.bg_roi.sigRegionChanged.connect(self.update_bg_label_pos)
        self.update_bg_label_pos()
        self.update_all_traces()

    def update_bg_label_pos(self):
        if self.bg_roi and self.bg_label:
            pos = self.bg_roi.pos()
            size = self.bg_roi.size()
            self.bg_label.setPos(pos.x() + size.x() / 2, pos.y() + size.y() / 2)

    def on_image_clicked(self, x, y):
        if self.stack is None or self.view_box.freehand_mode:
            return
        img_h, img_w = self.stack.shape[1:]
        if 0 <= x <= img_w and 0 <= y <= img_h:
            self.add_roi(center_x=x, center_y=y)

    def add_roi(self, center_x=None, center_y=None, pos=None, size=None):
        if self.stack is None:
            return
        h, w = self.stack.shape[1:]

        if pos and size:
            roi_pos = pos
            roi_size = size
        else:
            if center_x is None or center_y is None:
                center_x, center_y = w // 2, h // 2
            roi_pos = [center_x - 15, center_y - 15]
            roi_size = [30, 30]

        colors_list = ['r', 'g', 'b', 'c', 'm', 'y']
        color_code = colors_list[len(self.rois) % len(colors_list)]
        roi_num = len(self.rois) + 1

        circle_roi = pg.CircleROI(roi_pos, roi_size, pen=pg.mkPen(color_code, width=2))
        circle_roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        
        circle_roi.sigClicked.connect(self.on_roi_clicked)

        self.view_box.addItem(circle_roi)
        self.rois.append(circle_roi)

        roi_label = pg.TextItem(f"ROI {roi_num}", color=color_code, anchor=(0.5, 0.5))
        roi_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.view_box.addItem(roi_label)
        self.roi_labels.append(roi_label)

        circle_roi.sigRegionChanged.connect(self.update_all_traces)
        
        self.combo_roi_filter.blockSignals(True)
        self.combo_roi_filter.addItem(f"ROI {roi_num}")
        self.combo_roi_filter.blockSignals(False)

        self.btn_delete_selected_roi.setEnabled(True)
        self.update_all_traces()

    def on_roi_clicked(self, roi, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            menu = QMenu()
            delete_action = menu.addAction("🗑️ Delete ROI")
            action = menu.exec(ev.screenPos().toPoint())
            if action == delete_action:
                self.remove_roi(roi)

    def delete_selected_roi(self):
        if not self.rois:
            return

        filter_index = self.combo_roi_filter.currentIndex()
        if filter_index > 0:
            roi_to_delete = self.rois[filter_index - 1]
        else:
            roi_to_delete = self.rois[-1]

        self.remove_roi(roi_to_delete)

    def remove_roi(self, roi):
        if roi in self.rois:
            idx = self.rois.index(roi)
            self.view_box.removeItem(roi)
            self.view_box.removeItem(self.roi_labels[idx])

            del self.rois[idx]
            del self.roi_labels[idx]
            
            if idx in self.manual_peaks_override:
                del self.manual_peaks_override[idx]
            if idx in self.roi_custom_colors:
                del self.roi_custom_colors[idx]

            new_colors = {}
            for old_k, col_v in self.roi_custom_colors.items():
                if old_k < idx:
                    new_colors[old_k] = col_v
                elif old_k > idx:
                    new_colors[old_k - 1] = col_v
            self.roi_custom_colors = new_colors

            self.combo_roi_filter.blockSignals(True)
            self.combo_roi_filter.clear()
            self.combo_roi_filter.addItem("All ROIs")

            for i, (r_roi, label) in enumerate(zip(self.rois, self.roi_labels)):
                label.setText(f"ROI {i+1}")
                self.combo_roi_filter.addItem(f"ROI {i+1}")

            self.combo_roi_filter.setCurrentIndex(0)
            self.combo_roi_filter.blockSignals(False)

            if len(self.rois) == 0:
                self.btn_delete_selected_roi.setEnabled(False)

            self.on_active_trace_filter_changed()

    def clear_all_rois(self):
        for roi in list(self.rois):
            self.view_box.removeItem(roi)
        for label in list(self.roi_labels):
            self.view_box.removeItem(label)
        self.rois.clear()
        self.roi_labels.clear()
        self.roi_custom_colors.clear()

        self.combo_roi_filter.blockSignals(True)
        self.combo_roi_filter.clear()
        self.combo_roi_filter.addItem("All ROIs")
        self.combo_roi_filter.blockSignals(False)

    def on_bleach_mode_changed(self, idx):
        self.bleach_corr_mode = idx
        self.update_all_traces()

    def save_session(self):
        if self.stack is None:
            return

        default_save_path = os.path.join(self.last_dir, "calcium_session.json") if self.last_dir else "calcium_session.json"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Analysis Session", default_save_path, "Calcium Session (*.json)")
        if not file_path:
            return

        self.last_dir = os.path.dirname(file_path)

        clean_manual_peaks = {
            str(k): [int(p) for p in v] for k, v in self.manual_peaks_override.items()
        }

        clean_disabled_peaks = [list(p) for p in self.disabled_peaks]

        clean_cursors = {
            f"{k[0]}_{k[1]}": v for k, v in self.peak_cursors_map.items()
        }

        rois_serialized = []
        for r in self.rois:
            if isinstance(r, pg.PolyLineROI):
                pts = [r.mapToParent(h['item'].pos()) for h in r.handles]
                rois_serialized.append({
                    'type': 'poly',
                    'points': [[float(p.x()), float(p.y())] for p in pts]
                })
            else:
                rois_serialized.append({
                    'type': 'circle',
                    'pos': [float(r.pos().x()), float(r.pos().y())],
                    'size': [float(r.size().x()), float(r.size().y())]
                })

        session_data = {
            'video_path': self.current_stack_path,
            'ch2_path': self.stack_ch2_path,
            'ch3_path': self.stack_ch3_path,
            'frame_interval_ms': float(self.spin_frame_interval_ms.value()),
            'analysis_start': int(self.spin_analysis_start.value()),
            'analysis_end': int(self.spin_analysis_end.value()),
            'f0_start': int(self.spin_f0_start.value()),
            'f0_end': int(self.spin_f0_end.value()),
            'noise_mult': float(self.spin_noise_mult.value()),
            'min_duration': int(self.spin_min_duration.value()),
            'prominence': float(self.spin_prominence.value()),
            'distance': int(self.spin_distance.value()),
            'bleach_corr_mode': int(self.bleach_corr_mode),
            'apply_bleach_corr': bool(self.bleach_corr_mode > 0),
            'manual_peaks': clean_manual_peaks,
            'disabled_peaks': clean_disabled_peaks,
            'peak_cursors': clean_cursors,
            'roi_custom_colors': {str(k): v for k, v in self.roi_custom_colors.items()},
            'bg_roi': {
                'pos': [float(self.bg_roi.pos().x()), float(self.bg_roi.pos().y())],
                'size': [float(self.bg_roi.size().x()), float(self.bg_roi.size().y())]
            } if self.bg_roi else None,
            'rois': rois_serialized
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=4)
            QMessageBox.information(self, "Session Saved", "Analysis session successfully saved!")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Unable to save session:\n{str(e)}")

    def load_session(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Analysis Session", self.last_dir, "Calcium Session (*.json)")
        if not file_path or not os.path.exists(file_path):
            return

        self.last_dir = os.path.dirname(file_path)

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            v_path = data.get('video_path')
            if not v_path or not os.path.exists(v_path):
                QMessageBox.warning(self, "Video File Not Found", "Please select the primary video file for this session.")
                v_path, _ = QFileDialog.getOpenFileName(self, "Select Video", self.last_dir, "Compatible Files (*.tif *.tiff *.nd2 *.avi *.mp4)")
                if not v_path:
                    return

            if not self.load_file(v_path):
                return

            ch2_p = data.get('ch2_path')
            if ch2_p and os.path.exists(ch2_p):
                self.load_extra_channel(2, file_path=ch2_p)

            ch3_p = data.get('ch3_path')
            if ch3_p and os.path.exists(ch3_p):
                self.load_extra_channel(3, file_path=ch3_p)

            self.clear_all_rois()
            self.clear_all_overlays()

            self.spin_frame_interval_ms.setValue(data.get('frame_interval_ms', 100.0))
            self.spin_analysis_start.setValue(data.get('analysis_start', 1))
            self.spin_analysis_end.setValue(data.get('analysis_end', len(self.stack)))
            self.spin_f0_start.setValue(data.get('f0_start', 1))
            self.spin_f0_end.setValue(data.get('f0_end', 20))
            self.spin_noise_mult.setValue(data.get('noise_mult', 0.5))
            self.spin_min_duration.setValue(data.get('min_duration', 1))
            self.spin_prominence.setValue(data.get('prominence', 0.0001))
            self.spin_distance.setValue(data.get('distance', 1))
            
            if 'bleach_corr_mode' in data:
                self.bleach_corr_mode = int(data['bleach_corr_mode'])
            else:
                self.bleach_corr_mode = 2 if data.get('apply_bleach_corr', False) else 0
            self.combo_bleach_mode.setCurrentIndex(self.bleach_corr_mode)

            manual_p = data.get('manual_peaks', {})
            self.manual_peaks_override = {int(k): [int(p) for p in v] for k, v in manual_p.items()}

            disabled_p = data.get('disabled_peaks', [])
            self.disabled_peaks = {tuple(p) for p in disabled_p}

            loaded_cursors = data.get('peak_cursors', {})
            self.peak_cursors_map = {}
            for k, v in loaded_cursors.items():
                parts = k.split("_")
                self.peak_cursors_map[(int(parts[0]), int(parts[1]))] = v

            bg_data = data.get('bg_roi')
            if bg_data:
                self.set_bg_roi(pos=bg_data['pos'], size=bg_data['size'])

            for r_info in data.get('rois', []):
                if r_info.get('type') == 'poly' and 'points' in r_info:
                    self.add_freehand_roi(r_info['points'])
                else:
                    self.add_roi(pos=r_info['pos'], size=r_info['size'])

            saved_colors = data.get('roi_custom_colors', {})
            self.roi_custom_colors = {int(k): v for k, v in saved_colors.items()}
            for r_idx, col_hex in self.roi_custom_colors.items():
                if r_idx < len(self.rois):
                    self.rois[r_idx].setPen(pg.mkPen(col_hex, width=2))
                if r_idx < len(self.roi_labels):
                    self.roi_labels[r_idx].setColor(col_hex)

            self.update_all_traces()
            QMessageBox.information(self, "Session Loaded", "Analysis session successfully restored!")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Unable to restore session:\n{str(e)}")

    def update_roi_labels_positions(self):
        for roi, label in zip(self.rois, self.roi_labels):
            cx, cy = self._get_roi_center(roi)
            label.setPos(cx, cy)

    # ==============================================================================
    # ESTRAZIONE INTENSITÀ MEDIA ROI (SUPPORTA CIRCOLARI, RETTANGOLARI E POLIGONI)
    # ==============================================================================
    def get_roi_mean_trace(self, roi, start_frame, end_frame, target_stack=None):
        stk = target_stack if target_stack is not None else self.stack
        if stk is None:
            return np.zeros(max(0, end_frame - start_frame))

        safe_end = min(len(stk), end_frame)
        safe_start = min(safe_end, start_frame)
        sub_stack = stk[safe_start:safe_end]
        if len(sub_stack) == 0:
            return np.zeros(0)

        img_h, img_w = stk.shape[1:]

        if isinstance(roi, pg.PolyLineROI):
            pts = [roi.mapToParent(h['item'].pos()) for h in roi.handles]
            if len(pts) < 3:
                return np.zeros(len(sub_stack))
            poly_coords = np.array([[int(round(p.x())), int(round(p.y()))] for p in pts], dtype=np.int32)
            x_min = max(0, int(np.min(poly_coords[:, 0])))
            y_min = max(0, int(np.min(poly_coords[:, 1])))
            x_max = min(img_w, int(np.max(poly_coords[:, 0])) + 1)
            y_max = min(img_h, int(np.max(poly_coords[:, 1])) + 1)

            if x_max <= x_min or y_max <= y_min:
                return np.zeros(len(sub_stack))

            local_poly = poly_coords - np.array([x_min, y_min])
            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            cv2.fillPoly(mask, [local_poly], 255)

            mean_values = []
            for frame in sub_stack:
                sub_img = frame[y_min:y_max, x_min:x_max]
                if mask.shape == sub_img.shape:
                    pts_px = sub_img[mask > 0]
                    mean_values.append(np.mean(pts_px) if len(pts_px) > 0 else 0)
                else:
                    mean_values.append(0)
            return np.array(mean_values)

        pos = roi.pos()
        size = roi.size()
        x, y = int(pos.x()), int(pos.y())
        w, h = int(size.x()), int(size.y())

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img_w, x + w), min(img_h, y + h)

        if x1 <= x0 or y1 <= y0:
            return np.zeros(len(sub_stack))

        if isinstance(roi, pg.RectROI):
            mask = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
        else:
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)

        mean_values = []
        for frame in sub_stack:
            sub_img = frame[y0:y1, x0:x1]
            if mask.shape == sub_img.shape:
                pts = sub_img[mask > 0]
                mean_values.append(np.mean(pts) if len(pts) > 0 else 0)
            else:
                mean_values.append(0)

        return np.array(mean_values)

    def apply_bleaching_correction(self, trace, mode=1):
        """
        mode == 1: Gentle Linear Detrend (Mild)
        mode == 2: Exponential Decay Fit (1-Exp)
        mode == 3: Bi-Exponential Decay Fit (2-Exp: Fast + Slow)
        mode == 4: Tri-Exponential Decay Fit (3-Exp: Fast + Mid + Slow)
        """
        n = len(trace)
        if n < 4 or mode == 0:
            return trace

        t = np.arange(n, dtype=float)

        if mode == 1:
            # Gentle Linear Detrend anchored to initial baseline level
            seg_len = max(3, min(20, n // 5))
            start_base = np.percentile(trace[:seg_len], 30)
            end_base = np.percentile(trace[-seg_len:], 30)

            if end_base >= start_base:
                return trace

            slope = (end_base - start_base) / max(1.0, float(n - seg_len))
            drift = 0.75 * slope * t
            return trace - drift

        elif mode == 2:
            # Mono-Exponential Decay Fit anchored to initial trend[0]
            def exp_func(x, a, b, c):
                return a * np.exp(-b * x) + c

            try:
                popt, _ = curve_fit(
                    exp_func, t, trace,
                    p0=(max(0.0, trace[0] - trace[-1]), 0.001, trace[-1]),
                    bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]),
                    maxfev=2000
                )
                trend = exp_func(t, *popt)
                return trace - trend + trend[0]
            except Exception:
                return trace

        elif mode == 3:
            # Bi-Exponential Decay Fit (Fast + Slow) anchored to trend[0]
            def bi_exp_func(x, a1, b1, a2, b2, c):
                return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c

            amp_tot = max(0.0, float(trace[0] - trace[-1]))
            p0 = (
                0.6 * amp_tot,
                min(0.5, 5.0 / max(1.0, float(n))),
                0.4 * amp_tot,
                min(0.1, 0.5 / max(1.0, float(n))),
                float(trace[-1])
            )
            bounds = (
                [0.0, 0.0, 0.0, 0.0, -np.inf],
                [np.inf, 2.0, np.inf, 0.5, np.inf]
            )
            try:
                popt, _ = curve_fit(bi_exp_func, t, trace, p0=p0, bounds=bounds, maxfev=5000)
                trend = bi_exp_func(t, *popt)
                return trace - trend + trend[0]
            except Exception:
                return self.apply_bleaching_correction(trace, mode=2)

        elif mode == 4:
            # Tri-Exponential Decay Fit (Fast + Mid + Slow) anchored to trend[0]
            def tri_exp_func(x, a1, b1, a2, b2, a3, b3, c):
                return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + a3 * np.exp(-b3 * x) + c

            amp_tot = max(0.0, float(trace[0] - trace[-1]))
            p0 = (
                0.5 * amp_tot,
                min(0.8, 10.0 / max(1.0, float(n))),
                0.3 * amp_tot,
                min(0.2, 2.0 / max(1.0, float(n))),
                0.2 * amp_tot,
                min(0.05, 0.3 / max(1.0, float(n))),
                float(trace[-1])
            )
            bounds = (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf],
                [np.inf, 3.0, np.inf, 1.0, np.inf, 0.3, np.inf]
            )
            try:
                popt, _ = curve_fit(tri_exp_func, t, trace, p0=p0, bounds=bounds, maxfev=8000)
                trend = tri_exp_func(t, *popt)
                return trace - trend + trend[0]
            except Exception:
                return self.apply_bleaching_correction(trace, mode=3)

        return trace

    def _compute_channel_signals_for_roi(self, roi, target_stack, bg_trace, start_frame, end_frame, f0_s, f0_e):
        raw_trace = self.get_roi_mean_trace(roi, start_frame, end_frame, target_stack=target_stack)
        if len(raw_trace) == 0:
            return None

        if self.bleach_corr_mode > 0:
            raw_trace = self.apply_bleaching_correction(raw_trace, mode=self.bleach_corr_mode)

        if len(bg_trace) == len(raw_trace):
            sub_trace = raw_trace - bg_trace
        else:
            sub_trace = raw_trace

        safe_f0_e = min(len(sub_trace), f0_e)
        safe_f0_s = min(safe_f0_e, f0_s)
        if safe_f0_e <= safe_f0_s:
            safe_f0_e = min(len(sub_trace), safe_f0_s + 1)

        f0 = np.mean(sub_trace[safe_f0_s:safe_f0_e]) if safe_f0_e > safe_f0_s else np.mean(sub_trace)
        if f0 <= 0:
            f0 = 1e-6

        df_f0_trace = (sub_trace - f0) / f0
        return {
            'raw': raw_trace,
            'sub_bg': sub_trace,
            'df_f0': df_f0_trace,
            'f0_val': f0
        }

    # ==============================================================================
    # CALCOLO DELLE TRACCE E DEI PICCHI (MULTI-CANALE)
    # ==============================================================================
    def calculate_data(self):
        if self.stack is None or not self.rois:
            return None

        start_frame = max(0, self.spin_analysis_start.value() - 1)
        end_frame = min(len(self.stack), self.spin_analysis_end.value())
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        bg_trace = self.get_roi_mean_trace(self.bg_roi, start_frame, end_frame, target_stack=self.stack) if self.bg_roi is not None else np.zeros(end_frame - start_frame)
        bg_trace_ch2 = self.get_roi_mean_trace(self.bg_roi, start_frame, end_frame, target_stack=self.stack_ch2) if (self.bg_roi is not None and self.stack_ch2 is not None) else np.zeros(0)
        bg_trace_ch3 = self.get_roi_mean_trace(self.bg_roi, start_frame, end_frame, target_stack=self.stack_ch3) if (self.bg_roi is not None and self.stack_ch3 is not None) else np.zeros(0)
        
        f0_s = max(0, self.spin_f0_start.value() - 1 - start_frame)
        f0_e = min(end_frame - start_frame, self.spin_f0_end.value() - start_frame)
        if f0_e <= f0_s:
            f0_e = max(1, f0_s + 1)

        prominence = self.spin_prominence.value()
        distance = self.spin_distance.value()
        
        noise_mult = self.spin_noise_mult.value()
        min_duration = self.spin_min_duration.value()

        data_results = {'bg_trace': bg_trace, 'rois_data': []}

        for i, roi in enumerate(self.rois):
            ch1_sig = self._compute_channel_signals_for_roi(roi, self.stack, bg_trace, start_frame, end_frame, f0_s, f0_e)
            raw_trace = ch1_sig['raw']
            sub_trace = ch1_sig['sub_bg']
            df_f0_trace = ch1_sig['df_f0']
            f0 = ch1_sig['f0_val']
            analysis_trace = df_f0_trace

            if i in self.manual_peaks_override:
                peaks_df = np.array([p for p in self.manual_peaks_override[i] if 0 <= p < len(df_f0_trace)], dtype=int)
            else:
                median_val = np.median(analysis_trace)
                mad = np.median(np.abs(analysis_trace - median_val))
                roi_robust_std = mad * 1.4826
                abs_noise_threshold = noise_mult * roi_robust_std

                peaks_df, _ = find_peaks(
                    analysis_trace, 
                    height=abs_noise_threshold,
                    prominence=prominence, 
                    distance=distance,
                    width=min_duration if min_duration > 1 else None
                )

            roi_entry = {
                'name': f"ROI {i+1}", 'raw': raw_trace, 'sub_bg': sub_trace,
                'df_f0': df_f0_trace, 'f0_val': f0, 'peaks_indices': peaks_df
            }

            if self.stack_ch2 is not None:
                ch2_sig = self._compute_channel_signals_for_roi(roi, self.stack_ch2, bg_trace_ch2, start_frame, end_frame, f0_s, f0_e)
                if ch2_sig is not None:
                    roi_entry['ch2'] = ch2_sig

            if self.stack_ch3 is not None:
                ch3_sig = self._compute_channel_signals_for_roi(roi, self.stack_ch3, bg_trace_ch3, start_frame, end_frame, f0_s, f0_e)
                if ch3_sig is not None:
                    roi_entry['ch3'] = ch3_sig

            data_results['rois_data'].append(roi_entry)

        return data_results

    def update_all_traces(self):
        self.update_roi_labels_positions()
        self.update_bg_label_pos()

        data = self.calculate_data()
        if data is None:
            self.btn_export.setEnabled(False)
            self.btn_export_trace_img.setEnabled(False)
            self.btn_sync.setEnabled(False)
            self.btn_ecr_sync.setEnabled(False)
            self.btn_event_sync.setEnabled(False)
            self.btn_latency.setEnabled(False)
            self.btn_show_hub.setEnabled(False)
            self.btn_draw_cluster.setEnabled(False)
            self.btn_main_pdf_report.setEnabled(False)
            self.btn_go_peak_analysis.setEnabled(False)
            self.plot_widget.clear()
            return

        self.btn_export.setEnabled(True)
        self.btn_export_trace_img.setEnabled(len(self.rois) >= 1)
        self.btn_sync.setEnabled(len(self.rois) >= 2)
        self.btn_ecr_sync.setEnabled(len(self.rois) >= 2)
        self.btn_event_sync.setEnabled(len(self.rois) >= 2)
        self.btn_latency.setEnabled(len(self.rois) >= 1)
        self.btn_show_hub.setEnabled(len(self.rois) >= 2)
        self.btn_draw_cluster.setEnabled(len(self.rois) >= 3)
        self.btn_main_pdf_report.setEnabled(len(self.rois) >= 1)
        self.btn_go_peak_analysis.setEnabled(len(self.rois) >= 1)
        self.plot_widget.clear()

        mode_index = self.combo_mode.currentIndex()
        filter_index = self.combo_roi_filter.currentIndex()
        show_extra_ch = self.chk_overlay_multichannel.isChecked()

        num_pts = len(data['rois_data'][0]['df_f0']) if data['rois_data'] else 0
        x_axis, time_label = self.get_time_axis(num_pts)
        self.plot_widget.setLabel('bottom', time_label)

        for i, roi_info in enumerate(data['rois_data']):
            if filter_index > 0 and (filter_index - 1) != i:
                continue

            if mode_index == 0:
                plot_trace = roi_info['df_f0']
                self.plot_widget.setLabel('left', 'ΔF / F0')
            elif mode_index == 1:
                plot_trace = roi_info['sub_bg']
                self.plot_widget.setLabel('left', 'Intensity (RAW - BG)')
            else:
                plot_trace = roi_info['raw']
                self.plot_widget.setLabel('left', 'RAW Intensity')

            color = self._get_roi_display_color(i)
            ch1_label = f"{roi_info['name']} (Ch1)" if (show_extra_ch and ('ch2' in roi_info or 'ch3' in roi_info)) else roi_info['name']
            self.plot_widget.plot(x_axis, plot_trace, pen=pg.mkPen(color, width=2), name=ch1_label)

            if show_extra_ch and 'ch2' in roi_info:
                ch2_tr = roi_info['ch2']['df_f0'] if mode_index == 0 else (roi_info['ch2']['sub_bg'] if mode_index == 1 else roi_info['ch2']['raw'])
                x_ch2, _ = self.get_time_axis(len(ch2_tr))
                self.plot_widget.plot(
                    x_ch2, ch2_tr,
                    pen=pg.mkPen('#00e676', width=1.8, style=Qt.PenStyle.DashLine),
                    name=f"{roi_info['name']} (Ch2)"
                )

            if show_extra_ch and 'ch3' in roi_info:
                ch3_tr = roi_info['ch3']['df_f0'] if mode_index == 0 else (roi_info['ch3']['sub_bg'] if mode_index == 1 else roi_info['ch3']['raw'])
                x_ch3, _ = self.get_time_axis(len(ch3_tr))
                self.plot_widget.plot(
                    x_ch3, ch3_tr,
                    pen=pg.mkPen('#ff4081', width=1.8, style=Qt.PenStyle.DotLine),
                    name=f"{roi_info['name']} (Ch3)"
                )

            peaks = roi_info['peaks_indices']
            if len(peaks) > 0:
                peak_points = pg.ScatterPlotItem(
                    x=x_axis[peaks], y=plot_trace[peaks], symbol='o', size=9,
                    brush=pg.mkBrush(color), pen=pg.mkPen('w', width=1)
                )
                self.plot_widget.addItem(peak_points)

    def plot_sync_map(self):
        data = self.calculate_data()
        if data is None or len(data['rois_data']) < 2:
            return

        try:
            plt.close('all')
            num_rois = len(data['rois_data'])
            df_dict = {r['name']: r['df_f0'] for r in data['rois_data']}
            df_matrix = pd.DataFrame(df_dict)
            corr_matrix = df_matrix.corr()

            show_annot = True if num_rois <= 15 else False
            fig_size = max(8, min(16, num_rois * 0.35))
            font_size = max(6, min(10, 120 / num_rois))

            g = sns.clustermap(
                corr_matrix, 
                annot=show_annot, 
                fmt=".2f" if show_annot else "",
                annot_kws={"size": font_size} if show_annot else None,
                cmap=self.combo_cmap.currentText(), 
                vmin=-1, vmax=1, 
                linewidths=0.5 if num_rois <= 30 else 0,
                figsize=(fig_size, fig_size)
            )

            plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=font_size, rotation=90)
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=font_size, rotation=0)

            g.figure.tight_layout()

            ax_btn = g.fig.add_axes([0.82, 0.92, 0.16, 0.05])
            btn_exp = matplotlib.widgets.Button(ax_btn, '💾 Export Data')
            
            def on_click_exp(event):
                default_save_path = os.path.join(self.last_dir, "heatmap_correlation_matrix.xlsx") if self.last_dir else "heatmap_correlation_matrix.xlsx"
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Heatmap Correlation Matrix", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
                if file_path:
                    self.last_dir = os.path.dirname(file_path)
                    try:
                        reordered_idx = g.dendrogram_row.reordered_ind
                        reordered_matrix = corr_matrix.iloc[reordered_idx, reordered_idx]

                        if file_path.endswith('.xlsx'):
                            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                                corr_matrix.to_excel(writer, sheet_name='Original Correlation Matrix')
                                reordered_matrix.to_excel(writer, sheet_name='Clustered Matrix')
                        else:
                            corr_matrix.to_csv(file_path, decimal='.', sep=',')
                        QMessageBox.information(self, "Export Complete", f"Heatmap correlation matrix saved to:\n{file_path}")
                    except Exception as ex:
                        QMessageBox.critical(self, "Error", f"Unable to save file:\n{str(ex)}")

            btn_exp.on_clicked(on_click_exp)
            g.btn_ref = btn_exp

            g.fig.canvas.manager.set_window_title(f"Continuous Synchrony Map - {LEAD_AUTHOR}")
            g.fig.suptitle(f"Continuous ΔF/F0 Synchrony Map & Cluster ({num_rois} ROIs)", y=0.98, fontsize=12, fontweight='bold')
            plt.show(block=False)

        except Exception as e:
            QMessageBox.critical(self, "Heatmap Error", f"Unable to generate map:\n{str(e)}")

    def plot_ecr_sync_map(self):
        data = self.calculate_data()
        if data is None or len(data['rois_data']) < 2:
            return

        try:
            w_frames = self.spin_coincidence_window.value()

            peaks_dict = {}
            roi_names = []
            silent_rois = []

            for r in data['rois_data']:
                p_indices = r['peaks_indices']

                if len(p_indices) == 0:
                    silent_rois.append(r['name'])
                    continue

                peaks_dict[r['name']] = np.array(p_indices)
                roi_names.append(r['name'])

            num_active_rois = len(roi_names)

            if num_active_rois < 2:
                QMessageBox.warning(
                    self, "Insufficient Active ROIs", 
                    f"At least 2 active ROIs (with detected peaks) are required to build the event correlation map.\n\n"
                    f"• **Silent ROIs Excluded:** {', '.join(silent_rois) if silent_rois else 'None'}\n"
                    f"• **Active ROIs Found:** {num_active_rois}"
                )
                return

            ecr_matrix = np.zeros((num_active_rois, num_active_rois))

            for i in range(num_active_rois):
                peaks_A = peaks_dict[roi_names[i]]
                for j in range(num_active_rois):
                    if i == j:
                        ecr_matrix[i, j] = 1.0
                    else:
                        peaks_B = peaks_dict[roi_names[j]]
                        
                        coincidences_A = 0
                        for pa in peaks_A:
                            if np.any(np.abs(peaks_B - pa) <= w_frames):
                                coincidences_A += 1

                        coincidences_B = 0
                        for pb in peaks_B:
                            if np.any(np.abs(peaks_A - pb) <= w_frames):
                                coincidences_B += 1

                        denom = min(len(peaks_A), len(peaks_B))
                        coincidences = min(coincidences_A, coincidences_B)
                        ecr_matrix[i, j] = coincidences / denom if denom > 0 else 0.0

            df_ecr = pd.DataFrame(ecr_matrix, index=roi_names, columns=roi_names)

            plt.close('all')
            show_annot = True if num_active_rois <= 15 else False
            fig_size = max(8, min(16, num_active_rois * 0.35))
            font_size = max(6, min(10, 120 / num_active_rois))

            g = sns.clustermap(
                df_ecr, 
                annot=show_annot, 
                fmt=".2f" if show_annot else "",
                annot_kws={"size": font_size} if show_annot else None,
                cmap=self.combo_cmap.currentText(), 
                vmin=0, vmax=1, 
                linewidths=0.5 if num_active_rois <= 30 else 0,
                figsize=(fig_size, fig_size)
            )

            plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=font_size, rotation=90)
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=font_size, rotation=0)

            g.figure.tight_layout()

            ax_btn = g.fig.add_axes([0.82, 0.92, 0.16, 0.05])
            btn_exp = matplotlib.widgets.Button(ax_btn, '💾 Export Data')

            def on_click_exp(event):
                default_save_path = os.path.join(self.last_dir, "event_coincidence_rate_matrix.xlsx") if self.last_dir else "event_coincidence_rate_matrix.xlsx"
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Event Coincidence Rate", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
                if file_path:
                    self.last_dir = os.path.dirname(file_path)
                    try:
                        reordered_idx = g.dendrogram_row.reordered_ind
                        reordered_matrix = df_ecr.iloc[reordered_idx, reordered_idx]

                        if file_path.endswith('.xlsx'):
                            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                                df_ecr.to_excel(writer, sheet_name='ECR Matrix')
                                reordered_matrix.to_excel(writer, sheet_name='Clustered Matrix')
                        else:
                            df_ecr.to_csv(file_path, decimal='.', sep=',')
                        QMessageBox.information(self, "Export Complete", f"ECR matrix saved to:\n{file_path}")
                    except Exception as ex:
                        QMessageBox.critical(self, "Error", f"Unable to save file:\n{str(ex)}")

            btn_exp.on_clicked(on_click_exp)
            g.btn_ref = btn_exp

            dt_ms = self.spin_frame_interval_ms.value()
            w_ms = w_frames * dt_ms
            
            title_text = f"Event Coincidence Rate (ECR) (Window: ±{w_frames} frames / ±{w_ms:.1f}ms)"
            if silent_rois:
                title_text += f"\n🚫 Excluded Silent ROIs (0 Peaks): {', '.join(silent_rois)}"

            g.fig.canvas.manager.set_window_title(f"Event Coincidence Rate - {LEAD_AUTHOR}")
            g.fig.suptitle(title_text, y=0.98, fontsize=11, fontweight='bold')
            plt.show(block=False)

        except Exception as e:
            QMessageBox.critical(self, "Event Correlation Error", f"Unable to generate ECR matrix:\n{str(e)}")

    def plot_event_sync_map(self):
        data = self.calculate_data()
        if data is None or len(data['rois_data']) < 2:
            return

        try:
            start_f = max(0, self.spin_analysis_start.value() - 1)
            end_f = min(len(self.stack), self.spin_analysis_end.value())
            num_frames = end_f - start_f
            w_frames = self.spin_coincidence_window.value()

            bin_matrix = []
            roi_names = []
            silent_rois = []

            for r in data['rois_data']:
                p_indices = r['peaks_indices']

                if len(p_indices) == 0:
                    silent_rois.append(r['name'])
                    continue

                binary_vector = np.zeros(num_frames, dtype=float)
                for p in p_indices:
                    st = max(0, p - w_frames)
                    en = min(num_frames, p + w_frames + 1)
                    binary_vector[st:en] = 1.0

                bin_matrix.append(binary_vector)
                roi_names.append(r['name'])

            num_active_rois = len(roi_names)

            if num_active_rois < 2:
                QMessageBox.warning(
                    self, "Insufficient Active ROIs", 
                    f"At least 2 active ROIs (with detected peaks) are required to build the event correlation map.\n\n"
                    f"• **Silent ROIs Excluded:** {', '.join(silent_rois) if silent_rois else 'None'}\n"
                    f"• **Active ROIs Found:** {num_active_rois}"
                )
                return

            bin_matrix = np.array(bin_matrix)

            event_corr_matrix = np.zeros((num_active_rois, num_active_rois))

            for i in range(num_active_rois):
                for j in range(num_active_rois):
                    if i == j:
                        event_corr_matrix[i, j] = 1.0
                    else:
                        intersection = np.sum((bin_matrix[i] == 1.0) & (bin_matrix[j] == 1.0))
                        union = np.sum((bin_matrix[i] == 1.0) | (bin_matrix[j] == 1.0))
                        event_corr_matrix[i, j] = intersection / union if union > 0 else 0.0

            df_event_corr = pd.DataFrame(event_corr_matrix, index=roi_names, columns=roi_names)

            plt.close('all')
            show_annot = True if num_active_rois <= 15 else False
            fig_size = max(8, min(16, num_active_rois * 0.35))
            font_size = max(6, min(10, 120 / num_active_rois))

            g = sns.clustermap(
                df_event_corr, 
                annot=show_annot, 
                fmt=".2f" if show_annot else "",
                annot_kws={"size": font_size} if show_annot else None,
                cmap=self.combo_cmap.currentText(), 
                vmin=0, vmax=1, 
                linewidths=0.5 if num_active_rois <= 30 else 0,
                figsize=(fig_size, fig_size)
            )

            plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=font_size, rotation=90)
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=font_size, rotation=0)

            g.figure.tight_layout()

            ax_btn = g.fig.add_axes([0.82, 0.92, 0.16, 0.05])
            btn_exp = matplotlib.widgets.Button(ax_btn, '💾 Export Data')

            def on_click_exp(event):
                default_save_path = os.path.join(self.last_dir, "event_peak_coincidence_matrix.xlsx") if self.last_dir else "event_peak_coincidence_matrix.xlsx"
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Event-Based Coincidence", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
                if file_path:
                    self.last_dir = os.path.dirname(file_path)
                    try:
                        reordered_idx = g.dendrogram_row.reordered_ind
                        reordered_matrix = df_event_corr.iloc[reordered_idx, reordered_idx]

                        if file_path.endswith('.xlsx'):
                            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                                df_event_corr.to_excel(writer, sheet_name='Event Coincidence Matrix')
                                reordered_matrix.to_excel(writer, sheet_name='Clustered Matrix')
                        else:
                            df_event_corr.to_csv(file_path, decimal='.', sep=',')
                        QMessageBox.information(self, "Export Complete", f"Event coincidence matrix saved to:\n{file_path}")
                    except Exception as ex:
                        QMessageBox.critical(self, "Error", f"Unable to save file:\n{str(ex)}")

            btn_exp.on_clicked(on_click_exp)
            g.btn_ref = btn_exp

            dt_ms = self.spin_frame_interval_ms.value()
            w_ms = w_frames * dt_ms
            
            title_text = f"Discrete Event Coincidence Index (Jaccard) (Window: ±{w_frames} frames / ±{w_ms:.1f}ms)"
            if silent_rois:
                title_text += f"\n🚫 Excluded Silent ROIs (0 Peaks): {', '.join(silent_rois)}"

            g.fig.canvas.manager.set_window_title(f"Event Coincidence Index - {LEAD_AUTHOR}")
            g.fig.suptitle(title_text, y=0.98, fontsize=11, fontweight='bold')
            plt.show(block=False)

        except Exception as e:
            QMessageBox.critical(self, "Event Correlation Error", f"Unable to generate event correlation matrix:\n{str(e)}")

    def plot_activation_sequence(self):
        data = self.calculate_data()
        if data is None or not data['rois_data']:
            return

        dt_ms = self.spin_frame_interval_ms.value()
        activation_list = []
        for r in data['rois_data']:
            peaks = r['peaks_indices']
            first_peak_frame = peaks[0] if len(peaks) > 0 else np.nan
            first_peak_time_ms = peaks[0] * dt_ms if len(peaks) > 0 else np.nan
            activation_list.append({
                'ROI': r['name'], 
                'First_Peak_Frame': first_peak_frame,
                'First_Peak_Time_ms': first_peak_time_ms
            })

        df_act = pd.DataFrame(activation_list)
        df_active = df_act.dropna().sort_values(by='First_Peak_Time_ms').copy()

        if df_active.empty:
            QMessageBox.warning(self, "Warning", "No transient peaks detected in any ROI.")
            return

        min_time = df_active['First_Peak_Time_ms'].min()
        df_active['Relative_Latency_ms'] = df_active['First_Peak_Time_ms'] - min_time

        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            norm = plt.Normalize(df_active['First_Peak_Time_ms'].min(), df_active['First_Peak_Time_ms'].max())
            colors = plt.cm.plasma(norm(df_active['First_Peak_Time_ms']))

            bars = ax.barh(df_active['ROI'], df_active['First_Peak_Time_ms'], color=colors, edgecolor='black', linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel('Time of First Calcium Transient (Milliseconds)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Neuron ROI', fontsize=11, fontweight='bold')
            ax.set_title('Neuronal Activation Latency Sequence', fontsize=13, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + (df_active['First_Peak_Time_ms'].max() * 0.01), bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f} ms', va='center', ha='left', fontsize=9, fontweight='bold')

            plt.tight_layout()

            ax_btn = fig.add_axes([0.80, 0.91, 0.18, 0.05])
            btn_exp = matplotlib.widgets.Button(ax_btn, '💾 Export Latency')

            def on_click_exp(event):
                default_save_path = os.path.join(self.last_dir, "activation_latency.xlsx") if self.last_dir else "activation_latency.xlsx"
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Activation Latency Data", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
                if file_path:
                    self.last_dir = os.path.dirname(file_path)
                    try:
                        if file_path.endswith('.xlsx'):
                            df_active.to_excel(file_path, index=False)
                        else:
                            df_active.to_csv(file_path, index=False, decimal='.', sep=',')
                        QMessageBox.information(self, "Export Complete", f"Activation sequence data saved to:\n{file_path}")
                    except Exception as ex:
                        QMessageBox.critical(self, "Error", f"Unable to save file:\n{str(ex)}")

            btn_exp.on_clicked(on_click_exp)
            fig.btn_ref = btn_exp

            fig.canvas.manager.set_window_title("Activation Latency Map")
            plt.show(block=False)

        except Exception as e:
            QMessageBox.critical(self, "Latency Plot Error", f"Unable to generate plot:\n{str(e)}")

    def export_data(self):
        data = self.calculate_data()
        if data is None:
            return

        default_save_path = os.path.join(self.last_dir, "calcium_analysis_general.xlsx") if self.last_dir else "calcium_analysis_general.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save General Analysis Data", default_save_path, "Excel Workbook (*.xlsx);;CSV File (*.csv)")
        if not file_path:
            return

        self.last_dir = os.path.dirname(file_path)

        dt_ms = self.spin_frame_interval_ms.value()
        dt_sec = dt_ms / 1000.0
        num_frames = len(data['rois_data'][0]['df_f0']) if data['rois_data'] else 0
        total_time_sec = num_frames * dt_sec
        
        traces_dict = {
            'Frame': np.arange(num_frames),
            'Time (ms)': np.arange(num_frames) * dt_ms
        }
        for r in data['rois_data']:
            traces_dict[f"{r['name']}_Ch1_dF_F0"] = r['df_f0']
            if 'ch2' in r and len(r['ch2']['df_f0']) == num_frames:
                traces_dict[f"{r['name']}_Ch2_dF_F0"] = r['ch2']['df_f0']
            if 'ch3' in r and len(r['ch3']['df_f0']) == num_frames:
                traces_dict[f"{r['name']}_Ch3_dF_F0"] = r['ch3']['df_f0']

        df_traces = pd.DataFrame(traces_dict)

        summary_peaks_list = []
        for r in data['rois_data']:
            p_indices = r['peaks_indices']
            num_p = len(p_indices)
            freq_hz = num_p / total_time_sec if total_time_sec > 0 else 0
            
            if num_p > 1:
                iei_arr = np.diff(p_indices) * dt_sec
                mean_iei = np.mean(iei_arr)
            else:
                mean_iei = np.nan

            frames_str = ", ".join(map(str, p_indices)) if num_p > 0 else "None"

            summary_peaks_list.append({
                'ROI': r['name'],
                'Detected Peaks Count': int(num_p),
                'Firing Frequency (Hz)': float(freq_hz),
                'Mean Inter-Event Interval - IEI (s)': float(mean_iei) if not np.isnan(mean_iei) else np.nan,
                'Identified Peak Frames': frames_str
            })

        df_peaks_summary = pd.DataFrame(summary_peaks_list)

        try:
            if file_path.endswith('.xlsx'):
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df_traces.to_excel(writer, sheet_name='dF_F0 Traces', index=False)
                    df_peaks_summary.to_excel(writer, sheet_name='ROI Peak Summary', index=False)
            else:
                df_traces.to_csv(file_path, index=False, decimal='.', sep=',')

            QMessageBox.information(
                self, "Export Complete", 
                f"General report successfully saved to:\n{file_path}\n\n"
                f"• **Sheet 1:** Complete dF/F0 time series (including Ch2/Ch3 if loaded)\n"
                f"• **Sheet 2:** Peak counts, frequencies (Hz), and frame indices per ROI"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to save file:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalciumFluorescenceApp()
    window.show()
    sys.exit(app.exec())