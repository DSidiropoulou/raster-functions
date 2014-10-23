import numpy as np
import ctypes


class FocalStatistics():

    def __init__(self):
        self.name = "Focal Statistics"
        self.description = ""
        self.factor = 1.0
        self.emit = ctypes.windll.kernel32.OutputDebugStringA
        self.emit.argtypes = [ctypes.c_char_p]

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': ""
            },
            {
                'name': 'factor',
                'dataType': 'numeric',
                'value': 1.0,
                'required': True,
                'displayName': "Sampling Factor",
                'description': ""
            },
        ]


    def getConfiguration(self, **scalars):
        self.factor = scalars.get('factor', 1.0)
        return { 
            'samplingFactor': self.factor,
            'inputMask': True 
        }

        
    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['resampling'] = False
        kwargs['output_info']['cellSize'] = tuple(np.multiply(kwargs['raster_info']['cellSize'], self.factor))
        kwargs['output_info']['statistics'] = () 
        kwargs['output_info']['histogram'] = ()

        self.emit("Trace|FocalStatistics.UpdateRasterInfo|{0}\n".format(kwargs))
        return kwargs
        

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        p = pixelBlocks['raster_pixels']
        m = pixelBlocks['raster_mask']
        
        self.emit("Trace|Request Input Blocks|{0}\n".format(pixelBlocks['raster_pixels'].shape))
        
        # get pixel blocks
        sz = p.itemsize
        h,w = p.shape
        shapebl = (h/self.factor, w/self.factor, self.factor, self.factor)
        strides = sz*np.array([w*self.factor, self.factor, w, 1])
        blocks = np.lib.stride_tricks.as_strided(p, shape=shapebl, strides=strides)

        # get sum
        bstat=blocks.sum(axis=3)
        bstat=bstat.sum(axis=2)
        
        #average
        bstat=bstat/self.factor**2
        
        pixelBlocks['output_pixels'] = bstat.astype(props['pixelType'])     

        self.emit("Trace|Request Raster|{0}\n".format(props))
        self.emit("Trace|Request Size|{0}\n".format(shape))
        self.emit("Trace|Request Input Blocks|{0}\n".format(pixelBlocks['raster_pixels'].shape))
        self.emit("Trace|Request Blocks|{0}\n".format(pixelBlocks['output_pixels'].shape))
        return pixelBlocks

