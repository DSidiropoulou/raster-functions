import numpy as np
import ctypes


class FocalStatistics():

    def __init__(self):
        self.name = "Focal Statistics"
        self.description = ""
        self.factor = 1.0
        self.op = np.average
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
            {
                'name': 'op',
                'dataType': 'string',
                'value': 'Average',
                'required': False,
                'domain': ('Average', 'Max', 'Min'),
                'displayName': "Statistics",
                'description': "The type of statistic that will be calculated for the output pixel"
            },
        ]


    def getConfiguration(self, **scalars):
        self.factor = scalars.get('factor', 1.0)
        return { 
            'samplingFactor': self.factor,
            'inputMask': True 
        }

        
    def updateRasterInfo(self, **kwargs):
        s = kwargs.get('op', 'Average').lower()
        if s == 'average':
            self.op = np.average
        elif s == 'min': 
            self.op = np.min
        elif s == 'max': 
            self.op = np.max
               
        kwargs['output_info']['resampling'] = False
        kwargs['output_info']['cellSize'] = tuple(np.multiply(kwargs['raster_info']['cellSize'], self.factor))
        kwargs['output_info']['statistics'] = () 
        kwargs['output_info']['histogram'] = ()

        self.emit("Trace|FocalStatistics.UpdateRasterInfo|{0}\n".format(kwargs))
        return kwargs
        

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        p = pixelBlocks['raster_pixels']
        m = pixelBlocks['raster_mask']        
        
        # get pixel blocks
        sz = p.itemsize
        h,w = p.shape
        shapebl = (h/self.factor, w/self.factor, self.factor, self.factor)
        strides = sz*np.array([w*self.factor, self.factor, w, 1])
        blocks = np.lib.stride_tricks.as_strided(p, shape=shapebl, strides=strides)

        # get statistic
        bstat = self.op(blocks, axis=3)
        bstat = self.op(bstat, axis=2)        
      
        pixelBlocks['output_pixels'] = bstat.astype(props['pixelType'])      

        self.emit("Trace|Request Raster|{0}\n".format(props))
        self.emit("Trace|Request Size|{0}\n".format(shape))
       
        return pixelBlocks

