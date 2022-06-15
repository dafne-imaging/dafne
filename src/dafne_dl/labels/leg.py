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
    1: 'SOL',
    2: 'GM',
    3: 'GL',
    4: 'TA',
    5: 'ELD',
    6: 'PE',
}

long_labels = {
    1: 'Soleus',
    2: 'Gastrocnemius Medialis',
    3: 'Gastrocnemius Lateralis',
    4: 'Tibialis Anterior',
    5: 'Extensor Longus Digitorum',
    6: 'Peroneus'
}

long_labels_split = {}
ctr = 1
for key, val in long_labels.items():
    long_labels_split[ctr] = val + "_R"
    long_labels_split[ctr+1] = val + "_L"
    ctr += 2

inverse_labels = merge_dict(invert_dict(short_labels), invert_dict(long_labels))
