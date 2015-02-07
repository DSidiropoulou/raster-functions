import numpy as np
import ctypes

class StatusUpdate():

    def __init__(self):
        self.name = "Status Raster"
        self.description = ""
        self.emit = ctypes.windll.kernel32.OutputDebugStringA
        self.emit.argtypes = [ctypes.c_char_p]

    def getParameterInfo(self):
        return [
            {
                'name': 'classification',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Classified image",
                'description': ""
            },
            {
                'name': 'state',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Last state",
                'description': ""
            },
            {
                'name': 'new',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "New state",
                'description': ""
            }, 
           ]


    def getConfiguration(self, **scalars):        
        return {
            'inputMask': False,
        }

        
    def updateRasterInfo(self, **kwargs):
               
        kwargs['output_info']['bandCount'] = kwargs['new_info']['bandCount']
        kwargs['output_info']['resampling'] = False
          
        self.emit("Trace|FocalStatistics.UpdateRasterInfo|{0}\n".format(kwargs))
        return kwargs
        

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        status = pixelBlocks['state_pixels']
        new = pixelBlocks['new_pixels']

        c = pixelBlocks['classification_pixels']        
        
        indRU=np.where(c == 0) # ClassID / Color Index
        
        status[:,indRU[0], indRU[1]] = new[:,indRU[0], indRU[1]]

        # NoData cells of Status - update with new
##        indnd = np.where(status == 0)
##        status[indnd] = new[indnd]

        pixelBlocks['output_pixels'] = status.astype('u2')

        self.emit("Trace|Request Raster|{0}\n".format(props))
        self.emit("Trace|Request Size|{0}\n".format(shape))
               
        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        self.emit("Trace|Request KeyMetadata|{0}\n".format(keyMetadata))

        return keyMetadata

