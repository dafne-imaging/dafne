import numpy as np
from .pySplineInterp import SplineInterpROIClass
import functools
import matplotlib.pyplot as plt

def notify_parent_decorator(func):
    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        obj.maskpair_parent.invalidate_mask()
        return func(obj, *args, **kwargs)
    return wrapper

class SplineInterpWithNotification(SplineInterpROIClass):

    # decorate the methods that modify data
    addKnot = notify_parent_decorator(SplineInterpROIClass.addKnot)
    removeKnot = notify_parent_decorator(SplineInterpROIClass.removeKnot)
    replaceKnot = notify_parent_decorator(SplineInterpROIClass.replaceKnot)
    moveKnot = notify_parent_decorator(SplineInterpROIClass.moveKnot)
    removeAllKnots = notify_parent_decorator(SplineInterpROIClass.removeAllKnots)

    def __init__(self, maskpair_parent, smooth=False):
        SplineInterpROIClass.__init__(self, smooth)
        self.maskpair_parent = maskpair_parent


class RoiAndMaskPair:
    def __init__(self, mask_size):
        self.subroi_stack = None
        self.mask = None
        self.mask_size = mask_size

    # if the passed roi is not a member of the Spline-Notifier class, make it so
    def __wrap_roi(self, roi):
        if type(roi) != SplineInterpWithNotification:
            r = SplineInterpWithNotification(self, roi.smooth)
            # copy the properties to the new class
            for key,item in vars(roi).items():
                setattr(r, key, item)
        else:
            r = roi
        return r

    def clear_mask(self):
        self.mask = np.zeros(self.mask_size)
        self.invalidate_roi()

    def clear_subroi_representations(self):
        if not self.subroi_stack: return
        for r in self.subroi_stack:
            try:
                r.remove()
            except:
                pass

    def clear_subrois(self):
        if self.subroi_stack is None:
            self.subroi_stack = []
        self.clear_subroi_representations()
        self.subroi_stack = []
        self.invalidate_mask()

    def add_subroi(self, roi = None):
        self.mask_to_subroi() # make sure we have subrois. If the mask is not valid, this has no effect anyway
        if not roi:
            roi = SplineInterpWithNotification(self)
        r = self.__wrap_roi(roi)
        if self.subroi_stack is None:
            self.subroi_stack = []
        self.subroi_stack.append(r)
        self.invalidate_mask()
        return r

    def set_subroi(self, index, roi):
        self.mask_to_subroi()  # make sure we have subrois
        r = self.__wrap_roi(roi)
        try:
            self.subroi_stack[index].remove()
        except:
            pass
        self.subroi_stack[index] = r
        self.invalidate_mask()
        return r

    def delete_subroi(self, index):
        try:
            self.subroi_stack[index].remove()
        except:
            pass
        del self.subroi_stack[index]

    def set_subroi_stack(self, roi_stack):
        self.clear_subrois()
        for roi in roi_stack:
            self.subroi_stack.append(self.__wrap_roi(roi))
        # self.invalidate_mask() # this was already invalidated in clear_subrois

    def set_mask(self, mask):
        self.mask = mask.astype(np.uint8)
        self.invalidate_roi()

    def invalidate_roi(self):
        #print("Roi invalidated")
        self.clear_subroi_representations()
        self.subroi_stack = None

    def invalidate_mask(self):
        #print("Mask invalidated")
        self.mask = None

    def subroi_to_mask(self):
        if not self.mask_size: return
        if self.subroi_stack is None: return
        if self.mask is not None: return # do not recalculate mask if it is valid
        self.mask = np.zeros(self.mask_size, dtype=np.uint8)
        for subroi in self.subroi_stack:
            try:
                self.mask = np.logical_xor(self.mask, subroi.toMask(self.mask_size, False))
            except:
                pass
        self.mask = self.mask.astype(np.uint8)

    def mask_to_subroi(self):
        if self.mask is None: return
        if self.subroi_stack is not None: return # do not recalculate subrois if they are valid
        splineInterpList = SplineInterpWithNotification.FromMask(self.mask)  # run mask tracing
        #print(splineInterpList)
        if self.subroi_stack is None:
            self.subroi_stack = []
        else:
            for r in self.subroi_stack:
                try:
                    r.remove()
                except:
                    pass
            self.subroi_stack = []
        for roi in splineInterpList:
            self.subroi_stack.append(self.__wrap_roi(roi))
        #print("Mask to subroi", self.subroi_stack)

    """
        Synchronize masks and ROIs
    """
    def sync(self):
        if self.mask is None:
            self.subroi_to_mask()
        elif self.subroi_stack is None:
            self.mask_to_subroi()

    def get_mask(self):
        if self.mask is None:
            self.subroi_to_mask()
        return self.mask

    def get_subroi_stack(self):
        if self.subroi_stack is None:
            self.mask_to_subroi()
        return self.subroi_stack

    def get_subroi(self, index):
        return self.get_subroi_stack()[index]

    def get_subroi_len(self):
        stack = self.get_subroi_stack()
        if stack is None: # note that this is different than having zero length
            self.add_subroi()
        return len(stack)

class ROIManager:
    """
    A class to hold both ROIs and Masks and to switch dynamically from one to the other.
    The class keeps track of the modifications to the ROI/Mask and creates them when needed

    self.allROIs is a dict with the following structure: { roi_name: { image_number: RoiAndMaskPair ... } ... }

    """

    def __init__(self, mask_size):
        self.allROIs = {}
        self.mask_size = mask_size

    def is_empty(self):
        return not self.allROIs

    # generator to go through all rois and masks from all or a particular roi name/slice
    def all_rois_and_masks(self, roi_name = None, image_number = None):
        if roi_name is None:
            roi_iter = list(self.allROIs.keys())
        else:
            roi_iter = [roi_name]

        for roi_key in roi_iter:
            if image_number is None:
                image_iter = list(self.allROIs[roi_key].keys())
            else:
                image_iter = [int(image_number)]
            for image_key in image_iter:
                roi_and_mask = self.allROIs[roi_key][image_key]
                yield (roi_key, image_key), roi_and_mask

    def all_rois(self, roi_name = None, image_number = None):
        for key_tuple, roi_and_mask in self.all_rois_and_masks(roi_name, image_number):
            subroi_stack = roi_and_mask.get_subroi_stack()
            for subroi_index, subroi in enumerate(subroi_stack):
                yield (key_tuple[0], key_tuple[1], subroi_index), subroi

    def all_masks(self, roi_name = None, image_number = None):
        for key_tuple, roi_and_mask in self.all_rois_and_masks(roi_name, image_number):
            yield key_tuple, roi_and_mask.get_mask()

    # removes all the visual representations of the ROIs
    def clear(self, roi_name = None, image_number = None, only_clear_interface = False):
        if roi_name is None:
            roi_iter = list(self.allROIs.keys())
        else:
            roi_iter = [roi_name]

        for roi_key in roi_iter:
            if image_number is None:
                image_iter = list(self.allROIs[roi_key].keys())
            else:
                image_iter = [int(image_number)]
            for image_key in image_iter:
                if only_clear_interface:
                    self.allROIs[roi_key][image_key].clear_subroi_representations()
                else:
                    self.allROIs[roi_key][image_key].clear_subrois()
                    del self.allROIs[roi_key][image_key]
            if not only_clear_interface and not self.allROIs[roi_key]: del self.allROIs[roi_key] # if we removed all slices from a roi, delete the roi itself

    def clear_subroi(self, roi_name, image_number, subroi_number):
        self.allROIs[roi_name][image_number].delete_subroi(subroi_number)

    def get_roi_names(self):
        return list(self.allROIs.keys())

    def contains(self, roi_name, image_number = None):
        if roi_name not in self.allROIs:
            return False
        if image_number is not None:
            return image_number in self.allROIs[roi_name]
        else:
            return True

    def set_mask_size(self, mask_size):
        # this is a big deal, as we would need to resize all the masks. It might not make sense.
        # let's just pass it along for now, knowing that it won't probably work
        self.mask_size = mask_size
        for roi_stack in self.allROIs:
            for roi_and_mask in roi_stack:
                roi_and_mask.mask_size = mask_size

    def get_roi_mask_pair(self, roi_name, image_number) -> RoiAndMaskPair:
        image_number = int(image_number)
        if roi_name not in self.allROIs:
            self.allROIs[roi_name] = {}
        if image_number not in self.allROIs[roi_name]:
            self.allROIs[roi_name][image_number] = RoiAndMaskPair(self.mask_size)
        return self.allROIs[roi_name][image_number]

    # make sure that a roi exists for this slice, but only add a subroi if there is none
    def add_roi(self, roi_name, image_number):
        image_number = int(image_number)
        rm = self.get_roi_mask_pair(roi_name, image_number)
        if not rm.get_subroi_len():
            self.add_subroi(roi_name, image_number)

    def add_subroi(self, roi_name, image_number):
        image_number = int(image_number)
        rm = self.get_roi_mask_pair(roi_name, image_number)
        rm.add_subroi()

    def _get_set_roi(self, roi_name, image_number, subroi_number, newROI=None) -> SplineInterpWithNotification:
        image_number = int(image_number)
        # check that a ROI actually exists with this name for this slice
        rm = self.get_roi_mask_pair(roi_name, image_number)

        #print("RM found", rm)

        # check if the subroi number exists for this slice
        subroi_len = rm.get_subroi_len()
        if subroi_number < subroi_len:
            if newROI:
                newROI = rm.set_subroi(subroi_number, newROI)
                return newROI
            else:
                return rm.get_subroi(subroi_number)
        if subroi_len == 0:
            rm.add_subroi()
        # if it doesn't exist, check if last subroi of the desired slice is empty
        r = rm.get_subroi(-1)
        if len(r.knots) == 0:
            if newROI:
                newROI = rm.set_subroi(-1, newROI)
                return newROI
            else:
                return r

        # otherwise, make a new roi
        if newROI:
            newROI = rm.add_subroi(newROI)
            return newROI
        else:
            r = rm.add_subroi()
            return r

    def get_roi(self, roi_name, image_number, subroi_number=0) -> SplineInterpWithNotification:
        image_number = int(image_number)
        return self._get_set_roi(roi_name, image_number, subroi_number)

    def set_roi(self, roi_name, image_number, subroi_number, roi):
        image_number = int(image_number)
        return self._get_set_roi(roi_name, image_number, subroi_number, roi)

    def get_mask(self, roi_name, image_number) -> np.ndarray:
        image_number = int(image_number)
        return self.get_roi_mask_pair(roi_name, image_number).get_mask()

    def set_mask(self, roi_name, image_number, mask):
        image_number = int(image_number)
        self.get_roi_mask_pair(roi_name, image_number).set_mask(mask)

    def add_mask(self, roi_name, image_number):
        self.set_mask(roi_name, image_number, np.zeros(self.mask_size, dtype=np.uint8))

    def clear_mask(self, roi_name, image_number):
        self.add_mask(roi_name, image_number)

        
