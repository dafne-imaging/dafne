#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 09:26:07 2015

@author: francesco
"""

import matplotlib.pyplot as plt
try:
    import pydicom as dicom
except:
    import dicom
import numpy as np
import os
import sys
try:
    from utils.dicomUtils.misc import create_affine
except:
    from dicomUtils.misc import create_affine

#DEFAULT_INTERPOLATION = 'spline36'
DEFAULT_INTERPOLATION = None # DEBUG
INVERT_SCROLL = True

class ImageShow:
    
    contrastWindow = None
    channelBalance = np.array([1.0, 1.0, 1.0])
    
    def __init__(self, im = None, axes = None, window=None, cmap=None):
        ImageShow.contrastWindow = window
        
        self.imPlot = None
        
        if axes is None:
            #initialize the figure
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = axes
            self.fig = self.axes.get_figure()
            
        self.axes.axis('off')
        self.fig.canvas.mpl_connect('button_press_event', self.btnPressCB)
        self.fig.canvas.mpl_connect('button_release_event', self.btnReleaseCB)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouseMoveCB)
        self.fig.canvas.mpl_connect('scroll_event', self.mouseScrollCB)
        self.fig.canvas.mpl_connect('key_press_event', self.keyPressCB)
        self.fig.canvas.mpl_connect('key_release_event', self.keyReleaseCB)
            
        # stack of images
        self.imList = []
        self.dicomHeaderList = None
        self.curImage = None
        self.cmap = cmap
        self.isImageRGB = False
        self.basepath = ''
        self.basename = ''

        self.resolution = [1,1,1]
        self.affine = None
        self.transpose = None

        # These methods can be defined in a subclass and called when some event occurs
        #self.leftPressCB = None
        #self.leftMoveCB = None     
        #self.leftReleaseCB = None
        #self.refreshCB = None
        
        if im is not None:
            if type(im) is np.ndarray:
                print("Display array")
                self.loadNumpyArray(im)
            elif type(im) is str:
                if os.path.isdir(im):
                    self.loadDirectory(im)
                else:
                    try:
                        im = self.loadDicomFile(im)
                    except:
                        pass
                    self.imList.append(im)
            self.curImage = 0
            self.displayImage(0)
    
    def displayImageRGB(self):
        #print "Displaying image"
        dispImage = np.copy(self.image)
        dispImage[:,:,0] *= ImageShow.channelBalance[0]
        dispImage[:,:,1] *= ImageShow.channelBalance[1]
        dispImage[:,:,2] *= ImageShow.channelBalance[2]
        dispImage = (dispImage - ImageShow.contrastWindow[0])/(ImageShow.contrastWindow[1] - ImageShow.contrastWindow[0])
        dispImage[dispImage < 0] = 0
        dispImage[dispImage > 1] = 1
        if self.imPlot is None:
            self.imPlot = self.axes.imshow(dispImage, interpolation = DEFAULT_INTERPOLATION)
        else:
            self.imPlot.set_data(dispImage)
        self.redraw()
        
    
    def displayImage(self, im, cmap = None):
        if cmap is None:
          if self.cmap is None:
            cmap = 'gray'
          else:
            cmap = self.cmap

        try:
            oldSize = self.image.shape
        except:
            oldSize = (-1,-1)

        # im can be an integer index in the imList
        if isinstance(im, int):
            if im >= 0 and im < len(self.imList):
                self.curImage = im
                self.image = self.imList[im]
        else: # otherwise let's assume it is pixel data
            self.image = im

        title = ''
        try:
            title = self.instructions + '\n'
        except:
            pass
        
        title += 'Image: %d' % self.curImage
            
        try:
            self.axes.set_title(title)
        except:
            pass
            
            
        # calculate the contrast if it was not already defined
        if ImageShow.contrastWindow is None:
            ImageShow.contrastWindow = self.calcContrast(self.image)
        
        
        if self.image.ndim == 3:
            self.isImageRGB = True
            self.displayImageRGB()
            return
        else:
            self.isImageRGB = False
        
        self.setCmap(cmap)

        if self.imPlot:
            if oldSize != self.image.shape: # if the image shape is different, force a new imPlot to be created
                try:
                    self.imPlot.remove()
                except:
                    pass
                self.imPlot = None

        # Create the image plot if there is none; otherwise update the data in the existing frame (faster)
        if self.imPlot is None:
            self.imPlot = self.axes.imshow(self.image, interpolation = DEFAULT_INTERPOLATION, vmin=ImageShow.contrastWindow[0], vmax=ImageShow.contrastWindow[1], cmap=self.cmap, zorder = -1)
        else:
            self.imPlot.set_data(self.image)
            
        
    def redraw(self):
        try:
            self.refreshCB()
        except:
            pass
        self.fig.canvas.draw()
        
    def mouseScrollCB(self, event):
        step = -event.step if INVERT_SCROLL else event.step
        if self.curImage is None or (step > 0 and self.curImage == 0) or (step < 0 and self.curImage > len(self.imList)-1):
            return
            
        if event.inaxes != self.axes:
            return
            
        self.curImage = self.curImage - step;
        if self.curImage < 0:
            self.curImage = 0
        if self.curImage > len(self.imList)-1:
            self.curImage = len(self.imList) - 1
        self.displayImage(self.imList[int(self.curImage)], self.cmap)
        self.redraw()
        try:
            self.fig.canvas.setFocus()
        except:
            pass

    def keyReleaseCB(self, event):
        print("key release")
        pass

    def keyPressCB(self, event):
        event.step = 0
        if event.key == 'right' or event.key == 'down':
            event.step = 1 if INVERT_SCROLL else -1
        elif event.key == 'left' or event.key == 'up':
            event.step = -1 if INVERT_SCROLL else 1
        self.mouseScrollCB(event)

    def isCursorNormal(self):
        try:
            isCursorNormal = ( self.fig.canvas.cursor().shape() == 0 ) # if backend is qt, it gets the shape of the
                # cursor. 0 is the arrow, which means we are not zooming or panning.
        except:
            isCursorNormal = True
        return isCursorNormal

    def btnPressCB(self, event):
        if not self.isCursorNormal():
            #print("Zooming or panning. Not processing clicks")
            return
        if event.button == 1:
            try:
                self.leftPressCB(event)
            except:
                pass
        if event.button == 3:
            if event.dblclick:
                self.resetContrast()
            else:
                self.startContrast = ImageShow.contrastWindow
                self.startBalance = np.copy(ImageShow.channelBalance)
                self.startXY = (event.x, event.y)
                self.rightPressCB(event)
    
    def rightPressCB(self, event):
        pass
            
    def btnReleaseCB(self, event):
        if event.button == 1:
            try:
                self.leftReleaseCB(event)
            except:
                pass
        if event.button == 3:
            self.imPlot.set_interpolation(DEFAULT_INTERPOLATION)
            self.startXY = None # 
            self.redraw()
            self.rightReleaseCB(event)
        
    def rightReleaseCB(self, event):
        pass
        
    def resetContrast(self):
        ImageShow.contrastWindow = self.calcContrast(self.image)
        if not self.isImageRGB:
            self.imPlot.set_clim(ImageShow.contrastWindow)
            self.redraw()
        else:
            # if image is RGB, we need to redraw it completely. Maybe it will be too slow?
            self.displayImageRGB()
        
    # callback for mouse move
    def mouseMoveCB(self, event):
        if event.button == 1:
            try:
                self.leftMoveCB(event)
            except:
                pass
        if event.button != 3 or self.startXY is None:
            return
        
        self.imPlot.set_interpolation('none')
        # convert contrast limits into window center and size
        contrastCenter = (self.startContrast[0] + self.startContrast[1])/2
        contrastExtent = (self.startContrast[1] - self.startContrast[0])/2
        
        # calculate displacemente of the mouse
        xDisplacement = event.x - self.startXY[0]
        yDisplacement = event.y - self.startXY[1]

        if event.key == 'control':
            ImageShow.channelBalance[0] = self.startBalance[0] - (float(xDisplacement)/100 + float(yDisplacement)/100)
            ImageShow.channelBalance[1] = self.startBalance[1] + float(xDisplacement)/100
            ImageShow.channelBalance[2] = self.startBalance[2] + float(yDisplacement)/100
            ImageShow.channelBalance[ImageShow.channelBalance < 0] = 0
            ImageShow.channelBalance[ImageShow.channelBalance > 1] = 1
            #print ImageShow.channelBalance
        else:
            # recalculate the window
            # the displacements have negative sign because it feels more natural
            contrastCenter = contrastCenter - yDisplacement
            contrastExtent = contrastExtent - xDisplacement
            if contrastExtent < 1:
                contrastExtent = 1
            
            # set the contrast window
            ImageShow.contrastWindow = (contrastCenter - contrastExtent, contrastCenter + contrastExtent)
            
        if not self.isImageRGB:
            self.imPlot.set_clim(ImageShow.contrastWindow)
            self.redraw()
        else:
            # if image is RGB, we need to redraw it completely. Maybe it will be too slow?
            self.displayImageRGB()
    
    def calcContrast(self, im):
        maxVal = np.percentile(im, 90)
        if maxVal <= 1: maxVal = np.max(im.flat)
        return (0, maxVal) # stretch contrast to remove outliers
    
    def setCmap(self, cmap):
        self.cmap = cmap;
        if self.imPlot is not None:
            self.imPlot.set_cmap(cmap)
        
        self.redraw()
            
    def loadDicomFile(self, fname):
        print(fname)
        ds = dicom.read_file(fname)
        # rescale dynamic range to 0-4095
        try:
            pixelData = ds.pixel_array.astype(np.float32)
        except:
            ds.decompress()
            pixelData = ds.pixel_array.astype(np.float32)

        try:
            slThickness = ds.SpacingBetweenSlices
        except:
            slThickness = ds.SliceThickness

        ds.PixelData = ""
        self.dicomHeaderList.append(ds)

        self.resolution = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(slThickness)]
        return pixelData
        
    # append one image to the internal list
    def appendImage(self, im):
        if type(im) is str:
            try:
                im = self.loadDicomFile(im)
            except:
                print("Error loading file:", im)
                return
        self.imList.append(im)
        
    def loadNumpyArray(self, data):
        if np.max(data.flat) <= 1: data *= 1000
        
        #print data.shape
        for sl in range(data.shape[2]):
            self.appendImage(data[:,:,sl])
            
            
    # load a whole directory of dicom files
    def loadDirectory(self, path, nii_orientation = 'tra'):
        self.imList = []
        self.dicomHeaderList = None
        self.affine = None
        self.transpose = None
        dicom_ext = ['.dcm', '.ima']
        nii_ext = ['.nii', '.gz']
        npy_ext = ['.npy']
        path = os.path.abspath(path)
        _, ext = os.path.splitext(path)

        basename = os.path.basename(path)

        self.basename = basename
        self.fig.canvas.set_window_title(basename)

        if ext.lower() in nii_ext:
            # load nii
            import nibabel as nib
            niimage = nib.load(path)
            orig_affine = niimage.affine

            orig_orient = np.abs(orig_affine[0:3,0:3]).argmax(axis=1)[0:3] # get the primary axis orientations
            orig_signs = [1 if niimage.affine[i, orig_orient[i]] > 0 else -1 for i in range(len(niimage.shape))]
            # nii orientations are RAS (Right-Anterior-Superior). Radiological orientations are LPS (Left-Posterior-Superior)
            # orientations for correct display:
            # Tra: A-R-S+
            # Cor: S-R-A+
            # Sag: S-A+R-
            # orig_orient[0] -> axis corresponding to R, [1] == A, [2] == S
            if nii_orientation == 'tra':
                new_axes = [1, 0, 2]
                new_signs = [1, -1, 1]
            elif nii_orientation == 'cor':
                new_axes = [2, 0, 1]
                new_signs = [1, -1, 1]
            elif nii_orientation == 'sag':
                new_axes = [2, 1, 0]
                new_signs = [1, +1, -1]

            dataset = niimage.get_fdata()
            self.transpose = [ orig_orient[new_axes[ax]] for ax in range(3)]
            print(orig_affine)
            dataset = dataset.transpose([ orig_orient[new_axes[ax]] for ax in range(3)] )
            signs = [1,1,1]
            for ax in range(3):
                signs[ax] = orig_signs[ax]*new_signs[ax]
                if signs[ax] < 0:
                    dataset = np.flip(dataset, axis=ax)

            print("Signs/transpose:", self.transpose)

            #orients = np.array([(i,1 if niimage.affine[i,i]>0 else -1) for i in range(len(niimage.shape))])
            #dataset = niimage.as_reoriented(orients).get_fdata()
            if np.max(dataset) < 1:
                dataset *= 1000
            self.resolution = niimage.header.get_zooms()[0:3]
            self.resolution = [ self.resolution[self.transpose[ax]] for ax in range(3) ]
            print(self.resolution)
            self.transpose = [(1 + self.transpose[ax]) * signs[ax] for ax in range(3)]
            for sl in range(dataset.shape[2]):
                self.appendImage(dataset[:,:,sl])
            self.basepath = os.path.dirname(path)
            print(self.affine)

        elif ext.lower() in npy_ext:
            data = np.load(path).astype(np.float32)
            self.loadNumpyArray(data)
            self.basepath = os.path.dirname(path)
        else: # assume it's dicom
            self.transpose = [-2, 1, 3]  # the vertical direction is always flipped(?) between dicom and nii
            if os.path.isdir(path):
                basepath = path
            else: # dicom file is passed. load the containing directory
                basepath = os.path.dirname(path)
                self.fig.canvas.set_window_title(os.path.basename(basepath))

            self.basename = ''
            self.basepath = basepath
            for f in sorted(os.listdir(basepath)):
                if os.path.basename(f).startswith('.'): continue
                fname, ext = os.path.splitext(f)
                if ext.lower() in dicom_ext:
                    if self.dicomHeaderList is None: self.dicomHeaderList = []
                    self.appendImage(basepath + os.path.sep + f)
            self.affine = create_affine(self.dicomHeaderList)
            print(self.affine)
        if len(self.imList) > 0:
            self.curImage = 0
            self.displayImage(int(0))
                
                
            
# when called as a script, load all the images in the directory
if __name__ == "__main__":
    # test file
    #imFig = imageShow("image0001.dcm")
    #imFig.appendImage("image0002.dcm")
    imFig = ImageShow()
    imFig.loadDirectory(sys.argv[1])
    #imFig.loadDirectory('image0001.dcm')
    plt.show()
