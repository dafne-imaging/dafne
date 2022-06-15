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

from .utils import invert_dict, merge_dict

short_labels = {
    1: 'VL',
    2: 'VM',
    3: 'VI',
    4: 'RF',
    5: 'SAR',
    6: 'GRA',
    7: 'AM',
    8: 'SM',
    9: 'ST',
    10: 'BFL',
    11: 'BFS',
    12: 'AL'
}

long_labels = {
    1: 'Vastus Lateralis',
    2: 'Vastus Medialis',
    3: 'Vastus Intermedius',
    4: 'Rectus Femoris',
    5: 'Sartorius',
    6: 'Gracilis',
    7: 'Adductor Magnus',
    8: 'Semimembranosus',
    9: 'Semitendinosus',
    10: 'Biceps Femoris Long',
    11: 'Biceps Femoris Short',
    12: 'Adductor Longus'
}

long_labels_split = {}
ctr = 1
for key, val in long_labels.items():
    long_labels_split[ctr] = val + "_R"
    long_labels_split[ctr+1] = val + "_L"
    ctr += 2

inverse_labels = merge_dict(invert_dict(short_labels), invert_dict(long_labels))