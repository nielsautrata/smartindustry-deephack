import pptk
import numpy as np

xyz = pptk.rand(10, 3)

v = pptk.viewer(xyz)
v.set(point_size=0.5)
