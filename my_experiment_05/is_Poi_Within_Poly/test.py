import numpy as np
import build_map.load_map_data as lmd
import is_Poi_Within_Poly.is_Poi_WithIn_Poly as ipwp
import transition_probability.load_tp_data as ltd


#  测试点是否在指定六边形内
def test_1():
    x = lmd.load_map_data().loc[1000, 1::2].values
    y = lmd.load_map_data().loc[1000, 2::2].values
    data = [list(item) for item in zip(x, y)]

    print(ipwp.is_Poi_Within_Poly([103.751, 30.5231], data))



