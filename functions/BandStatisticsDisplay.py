import numpy as np
import ctypes
from datetime import datetime
from calendar import monthrange


class BandStatisticsDisplay():

    def __init__(self):
        self.name = "Band Statistics"
        self.description = ""
        self.factor = 1
        self.emit = ctypes.windll.kernel32.OutputDebugStringA
        self.emit.argtypes = [ctypes.c_char_p]

    def getParameterInfo(self):
        return [
            {
                'name': 'mltsp',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Multispectral Image",
                'description': ""
            },
            {
                'name': 'factor',
                'dataType': 'numeric',
                'value': 1,
                'required': True,
                'displayName': "Sampling Factor",
                'description': ""
            },
        ]


    def getConfiguration(self, **scalars):
        self.factor = scalars.get('factor', 1)
        return {
            'inputMask': False,
            'keyMetadata': ('stdtime', 'acquisitiondate'),
            }

        
    def updateRasterInfo(self, **kwargs):
       
        kwargs['output_info']['bandCount'] = 3 
        kwargs['output_info']['resampling'] = True

        self.emit("Trace|FocalStatistics.UpdateRasterInfo|{0}\n".format(kwargs))
        return kwargs
        

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        m = np.array(pixelBlocks['mltsp_pixels'], dtype='f4')  # read only the necessary bands
       
        np.seterr(divide='ignore')
       
        # NDVI Calculation
        ndvi = (m[2] - m[1]) / (m[2] + m[1])
        ndvi = (ndvi * 32768) + 32768

        # NDBI Calculation
        ndbi = (m[0] - m[2]) / (m[0] + m[2]) + (m[0] - m[3]) / (m[0] + m[3])
        ndbi = (ndbi * 32768) + 32768

        # NDWI Calculation
        ndwi = (m[0] - m[3]) / (m[0] + m[3])
        ndwi = (ndwi * 32768) + 32768       

        # Create stack 
        ndvi = ndvi.reshape(1, ndvi.shape[0], ndvi.shape[1])
        ndbi = ndbi.reshape(1, ndbi.shape[0], ndbi.shape[1])
        ndwi = ndwi.reshape(1, ndwi.shape[0], ndwi.shape[1])
        m = np.vstack((ndvi, ndbi, ndwi))
        
       
        # Pixel blocks
        sz = m.itemsize 
        h = m.shape[len(m.shape)-2]
        w = m.shape[len(m.shape)-1]
        shapebl = (m.shape[0], h/self.factor, w/self.factor, self.factor, self.factor) 
        strides = sz*np.array([h*w, w*self.factor, self.factor, w, 1])
        blocks = np.lib.stride_tricks.as_strided(m, shape=shapebl, strides=strides)        
        axes = (len(blocks.shape)-2, len(blocks.shape)-1)
                       
        # Statistic
        avr = np.average(blocks, axes)

        # Copy average value factor x factor times
        avr = np.repeat(avr, self.factor, axis = avr.ndim-2)
        avr = np.repeat(avr, self.factor, axis = avr.ndim-1)
        
        # Replace values in the original raster, use original values along boundaries
        m[0:avr.shape[0], 0:avr.shape[1], 0:avr.shape[2]] = avr
        
        pixelBlocks['output_pixels'] = m.astype(props['pixelType'])

        self.emit("Trace|Request Raster|{0}\n".format(props))
        self.emit("Trace|Request Size|{0}\n".format(shape))
               
        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        self.emit("Trace|Request KeyMetadata|{0}\n".format(keyMetadata))

        return keyMetadata

