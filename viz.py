import numpy as np
import open3d as o3d

pts = np.load("pcd.npy")
f = open("test.xyzrgb", "w")
for p in pts:
  f.write("%.3f %.3f %.3f %.3f %.3f %.3f\n"%(p[3], p[4], p[5], p[0]/255, p[1]/255, p[2]/255))
f.close()

pcd = o3d.io.read_point_cloud("test.xyzrgb")
o3d.visualization.draw_geometries([pcd])
