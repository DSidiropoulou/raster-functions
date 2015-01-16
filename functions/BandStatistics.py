import numpy as np
import ctypes
from datetime import datetime
from calendar import monthrange


class BandStatistics():

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
                'name': 'pan',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Panchromatic Image",
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
            'samplingFactor': self.factor,
            'inputMask': False,
            'keyMetadata': ('stdtime', 'acquisitiondate'),
            }

        
    def updateRasterInfo(self, **kwargs):
       
        kwargs['output_info']['bandCount'] = 5 
        kwargs['output_info']['resampling'] = False
        kwargs['output_info']['cellSize'] = tuple(np.multiply(kwargs['mltsp_info']['cellSize'], self.factor))
        kwargs['output_info']['statistics'] = () 
        kwargs['output_info']['histogram'] = ()     

        self.emit("Trace|FocalStatistics.UpdateRasterInfo|{0}\n".format(kwargs))
        return kwargs
        

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        m = np.array(pixelBlocks['mltsp_pixels'][2:6], dtype='f4')  # read only the necessary bands
        p = pixelBlocks['pan_pixels']
        
        np.seterr(divide='ignore')

       
        # NDVI Calculation
        ndvi = (m[2] - m[1]) / (m[2] + m[1])
        ndvi = (ndvi * 32768) + 32768

        # NDBI Calculation
        ndbi = (m[3] - m[2]) / (m[3] + m[2])
        ndbi = (ndbi * 32768) + 32768

        # NDBI Calculation
        ndwi = (m[0] - m[3]) / (m[0] + m[3])
        ndwi = (ndwi * 32768) + 32768        

        # Create stack 
        m[0] = ndvi
        m[1] = ndbi
        m[2] = ndwi
        m[3] = p

       
        # Pixel blocks
        sz = m.itemsize 
        h = m.shape[len(m.shape)-2]
        w = m.shape[len(m.shape)-1]
        shapebl = (m.shape[0], h/self.factor, w/self.factor, self.factor, self.factor) 
        strides = sz*np.array([h*w, w*self.factor, self.factor, w, 1])
        blocks = np.lib.stride_tricks.as_strided(m, shape=shapebl, strides=strides)        
        axes = (len(blocks.shape)-2, len(blocks.shape)-1)
        axesp = (len(blocks.shape)-3, len(blocks.shape)-2)

               
        # Statistic   
        avr = np.average(blocks, axes)
        std = np.zeros((1, blocks.shape[1], blocks.shape[2]))
        std[0] = np.std(blocks[3], axesp)  # Index of pan band in the stack
                        
        pixelBlocks['output_pixels'] = np.vstack((avr, std)).astype(props['pixelType'])

        self.emit("Trace|Request Raster|{0}\n".format(props))
        self.emit("Trace|Request Size|{0}\n".format(shape))
               
        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        self.emit("Trace|Request KeyMetadata|{0}\n".format(keyMetadata))

        return keyMetadata

