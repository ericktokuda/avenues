import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct

_lacunarity = npct.load_library('liblacunarity', './src/')

def define_arguments():
    ''' Convenience function for defining the arguments of the functions
        inside the imported module. '''

    npflags = ['C_CONTIGUOUS']   # Requires a C contiguous array in memory
    float_2d_type = npct.ndpointer(dtype=np.float32, ndim=2, flags=npflags)
    float_1d_type = npct.ndpointer(dtype=np.float32, ndim=1, flags=npflags)

    args = [float_2d_type,
			float_1d_type,
            ct.c_int,      # Integer type
            ct.c_int,
            ct.c_int,
			ct.c_int

    ]
    _lacunarity.lacunarity.argtypes = args
    _lacunarity.lacunarity.restype = ct.c_int
    
define_arguments()

def lacunarity(img, max_radius, delta_radius=2):
	"""Calculate the lacunarity [1] of a binary image. The object in the image must have
	value 1 while the background must have value 0.
	The function returns the lacunarity calculated at the following radii:
	
	radii = np.arange(1, max_radius, delta_radius)

	[1] Rodrigues, E.P., Barbosa, M.S. and Costa, L.D.F., 2005. Self-referred approach to lacunarity. 
	    Physical Review E, 72(1), p.016707.

	Parameters
	----------
	img : np.ndarray
		Binary image to calculate the lacunarity.
	max_radius : int
		Maximum radius (scale) to use for lacunarity.
	delta_radius : int
		Radius increment between scales.

	Returns
	-------
	radii : np.ndarray
		Radius values used for lacunarity calculation
	lacunarity_values : np.ndarray
		Lacunarity values
	"""

	if img.dtype!=np.float32:
		img = img.astype(np.float32)

	radii = np.arange(1, max_radius, delta_radius)

	size_y, size_x = img.shape
	img = img.astype(np.float32)
	lacunarity_values = np.zeros(radii.size, dtype=np.float32)
	_lacunarity.lacunarity(img, lacunarity_values, size_y, size_x, max_radius, delta_radius)	
		
	return radii, lacunarity_values
