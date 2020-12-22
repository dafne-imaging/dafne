from matplotlib.patches import Circle, Polygon
from utils.pySplineInterp import SplineInterpROIClass

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
        self.painted = False
        if axes:
            axes.patches = []
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
        if clear_first:
            self.clear_patches()
        if self.painted: return
        for knot in self._knots:
            axes.add_patch(knot)
        for curve in self._curves:
            axes.add_patch(curve)
        self.painted = True


