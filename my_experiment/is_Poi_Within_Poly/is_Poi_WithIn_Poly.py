'''
判断一个点是否在六边形区域内
'''


def is_Ray_Intersects_Segment(poi, s_poi, e_poi):  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
        return False

    xseg = e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True   # 排除上述情况之后


def is_Poi_Within_Poly(poi, poly):
    # 可以先判断点是否在外包矩形内
    # if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc = 0  # 交点个数
    for i in range(poly.__len__()):  # [0,len-1]
        if i != poly.__len__()-1:
            s_poi = poly[i]
            e_poi = poly[i + 1]
        else:
            s_poi = poly[i]
            e_poi = poly[0]
        if is_Ray_Intersects_Segment(poi, s_poi, e_poi):
            sinsc += 1  # 有交点就加1

    return True if sinsc % 2 == 1 else False