import numpy as np
import ctypes


class StateChange():

    def __init__(self):
        self.name = "State Change"
        self.description = ""
        self.emit = ctypes.windll.kernel32.OutputDebugStringA
        self.emit.argtypes = [ctypes.c_char_p]

    def getParameterInfo(self):
        return [
            {
                'name': 'status',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Reference State",
                'description': ""
            },
            {
                'name': 'new',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "New State",
                'description': ""
            },
        ]

    def getConfiguration(self, **scalars):
        return { 
            'resampling': True,

        } 
      
    def updateRasterInfo(self, **kwargs):
        
        kwargs['output_info']['bandCount'] = 7 # NUMBER OF OUPUT BANDS
        kwargs['output_info']['resampling'] = False
        kwargs['output_info']['statistics'] = () 
        kwargs['output_info']['histogram'] = ()

        self.emit("Trace|FocalStatistics.UpdateRasterInfo|{0}\n".format(kwargs))
        return kwargs
        

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        status = pixelBlocks['status_pixels']
        new = pixelBlocks['new_pixels']

        diff = new-status
        diff = diff/2 + 32768
        statbands = status[1:3]

        stat = np.vstack((diff, statbands))

        pixelBlocks['output_pixels'] = stat.astype(props['pixelType'])

        self.emit("Trace|Request Raster|{0}\n".format(props))
        self.emit("Trace|Request Size|{0}\n".format(shape))
               
        return pixelBlocks

