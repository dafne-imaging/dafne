#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from matplotlib.patches import Circle, Polygon
from ..utils.pySplineInterp import SplineInterpROIClass

MIN_KNOT_RADIUS = 0.5 # if radius is smaller than this, knots are not painted


class ContourPainter:
    """
    Class to paint a series of ROIs on a pyplot axes object
    """
    def __init__(self, color, knot_radius):
        self._knots = []
        self._curves = []
        self.rois = []
        self.color = color
        self.knot_radius = knot_radius
        self.painted = False

    def set_color(self, color):
        self.color = color
        self.recalculate_patches()

    def set_radius(self, radius):
        self.radius = radius
        self.recalculate_patches()

    def clear_patches(self, axes = None):
        if not self.painted: return
        self.painted = False
        if axes:
            # axes.patches = [] # Error with new matplotlib!
            while axes.patches:
                axes.patches.pop()
            return
        for knot in self._knots:
            try:
                knot.set_visible(False)
            except:
                pass
            try:
                knot.remove()
            except:
                pass
        for curve in self._curves:
            try:
                curve.set_visible(False)
            except:
                pass
            try:
                curve.remove()
            except:
                pass

    def recalculate_patches(self):
        self.clear_patches()
        self._knots = []
        self._curves = []
        for roi in self.rois:
            if self.knot_radius >= MIN_KNOT_RADIUS:
                for knot in roi.knots:
                    self._knots.append(Circle(knot,
                                              self.knot_radius,
                                              facecolor='none',
                                              edgecolor=self.color,
                                              linewidth=1.0))
            try:
                self._curves.append(Polygon(roi.getCurve(),
                                        facecolor = 'none',
                                        edgecolor = self.color,
                                        zorder=1))
            except:
                pass

    def add_roi(self, roi: SplineInterpROIClass):
        self.rois.append(roi)
        self.recalculate_patches()

    def clear_rois(self, axes = None):
        self.clear_patches(axes)
        self._knots = []
        self._curves = []
        self.rois = []

    def draw(self, axes, clear_first=False):
        # print("Calling Contourpainter draw")
        if clear_first:
            self.clear_patches()
        for knot in self._knots:
            #if not self.painted:
            axes.add_patch(knot)
            axes.draw_artist(knot)
            self.painted = True
        for curve in self._curves:
            #if not self.painted:
            axes.add_patch(curve)
            axes.draw_artist(curve)
            self.painted = True
        # print("Painted?", self.painted)



