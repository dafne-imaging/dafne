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

from matplotlib.patches import Polygon, Rectangle
import numpy as np
import math
from scipy.ndimage import shift


class SquareBrush(Rectangle):

    def __init__(self, *args, **kwargs):
        Rectangle.__init__(self, *args, **kwargs)

    def to_mask(self, shape):
        """
        convert the brush to a binary mask of the size "shape"
        """
        mask = np.zeros(shape, dtype=np.uint8)

        xy = self.get_xy()
        h = self.get_height()
        w = self.get_width()
        # x and y are inverted
        x0 = int(np.round(xy[1] + 0.5))
        y0 = int(np.round(xy[0] + 0.5))
        x1 = int(x0 + np.round(h))
        y1 = int(y0 + np.round(w))

        mask[x0:x1, y0:y1] = 1
        return mask


class PixelatedCircleBrush(Polygon):

    def __init__(self, center, radius, **kwargs):
        self.point_array = None
        self.kwargs = kwargs
        self.center = None
        self.radius = None
        self.base_mask = None
        Polygon.__init__(self, np.array([[0,0],[1,1]]), **kwargs)
        self.center = np.array(center).ravel()  # make sure it's a row vector
        self.set_radius(radius)

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.radius

    def set_center(self, center):
        self.center = np.array(center).ravel() # make sure it's a row vector
        self._recalculate_xy()

    def set_radius(self, radius):
        if radius != self.radius:
            self.radius = radius
            self._recalculate_vertices()
            self._recalculate_mask()
            self._recalculate_xy()

    def to_mask(self, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[0:self.base_mask.shape[0], 0:self.base_mask.shape[1]] = self.base_mask
        mask = shift(mask, (self.center[1] - self.radius, self.center[0] - self.radius), order=0, prefilter=False)
        return mask

    def _recalculate_xy(self):
        xy = self.point_array + self.center
        self.set_xy(xy)

    def _recalculate_vertices(self):
        if self.radius == 0:
            self.point_array = np.array([[0,0],[1,0],[1,1],[0,1]]) - 0.5
            return

        radius = self.radius + 0.5

        # midpoint circle algorithm
        x = radius
        y = 0
        P = 1 - radius

        octant_point_array = [(x, y-0.5)]


        while x > y:

            y += 1

            # Mid-point inside or on the perimeter
            if P <= 0:
                P = P + 2 * y + 1

            # Mid-point outside the perimeter
            else:
                octant_point_array.append((x, y-0.5))
                x -= 1
                octant_point_array.append((x, y-0.5))
                P = P + 2 * y - 2 * x + 1

            if x < y:
                break

        # assemble the octants
        quarter_point_array = octant_point_array[:]
        quarter_point_array.extend([(y,x) for x,y in octant_point_array[::-1]])
        point_array = quarter_point_array[:]
        point_array.extend([(-x,y) for x,y in quarter_point_array[::-1]])
        point_array.extend([(-x,-y) for x,y in quarter_point_array])
        point_array.extend(([(x,-y) for x,y in quarter_point_array[::-1]]))
        
        self.point_array = np.array(point_array)

    def _recalculate_mask(self):
        if self.radius == 0:
            self.base_mask = np.ones((1, 1), dtype=np.uint8)
            return

        radius = self.radius

        # midpoint circle algorithm
        x = radius
        y = 0
        P = 1 - radius

        self.base_mask = np.zeros((self.radius * 2 + 1, self.radius * 2 + 1), dtype=np.uint8)

        def fill_mask_line(x, y):
            r = radius
            self.base_mask[int(r - x): int(r + x + 1), int(r + y)] = 1
            self.base_mask[int(r - x):int(r + x + 1), int(r - y)] = 1

        fill_mask_line(x, y)
        fill_mask_line(y, x)

        while x > y:
            y += 1

            # Mid-point inside or on the perimeter
            if P <= 0:
                P = P + 2 * y + 1

            # Mid-point outside the perimeter
            else:
                x -= 1
                P = P + 2 * y - 2 * x + 1

            fill_mask_line(x, y)
            fill_mask_line(y, x)

            if (x < y):
                break