'''
空车转移矩阵
'''
import numpy as np
import pandas as pd
import build_map.load_map_data as lmd
import is_Poi_Within_Poly.is_Poi_WithIn_Poly as ipwip
import transition_probability.load_tp_data as ltd


#  将数据中的六边形ID转为六边形中心坐标
def new_transition_probability():
    mapData = lmd.load_map_data()  # 导入转移概率数据中每个格子对应的六个顶点经纬度
    tpData = ltd.load_tp_data().values  # 导入空车转移概率
    mapData = mapData.iloc[:, 0:3:1].values  # 取六边形ID，中点经纬度

    # 将六边形数据转为字典
    map_Dict = dict()

    for i in range(len(mapData)):
        map_Dict.setdefault(mapData[i][0], []).append(mapData[i][1])
        map_Dict.setdefault(mapData[i][0], []).append(mapData[i][2])

    startPos_x = []
    startPos_y = []
    endPos_x = []
    endPos_y = []
    removeRow = []

    print(len(tpData))
    # 如果空车转移概率小于
    for a in range(len(tpData)):
        if not map_Dict.__contains__(tpData[a][1]) or not map_Dict.__contains__(tpData[a][2]):
            removeRow.append(a)
        else:
            startpos = map_Dict.get(tpData[a][1])
            startPos_x.append(startpos[0])
            startPos_y.append(startpos[1])
            endpos = map_Dict.get(tpData[a][2])
            endPos_x.append(endpos[0])
            endPos_y.append(endpos[1])

    print("需要移除的行有: {}".format(len(removeRow)))
    # print("removeRow中有 : {}".format(removeRow))
    for i in range(len(removeRow)):
        tpData = np.delete(tpData, removeRow[i], axis=0)

    print("移除问题行后，tpData还剩 : {}".format(len(tpData)) + "行")
    print("startPos_x 的长度为{}, startPos_y 的长度为{}".format(len(startPos_x), len(startPos_y)))
    print("endPos_x 的长度为{}, endPos_y 的长度为{}".format(len(endPos_x), len(endPos_y)))
    # 将数据格式由0-23小时改为1-24小时
    time = tpData[:, 0]
    for t in range(len(time)):
        if time[t] == 0:
            time[t] = 24
    # 打包新数据
    data = [list(item) for item in zip(time, startPos_x, startPos_y, endPos_x, endPos_y, tpData[:, 3])]
    data = np.array(data)
    print(data.shape)
    # 将数据导入new_transition_probability
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    np.savetxt("/Users/fangzhipeng/Desktop/new_transition_probability", data, fmt='%.06f', delimiter=',')


def lstm_transition_data():
    path = '../dataset/idle_transition_probability'
    data = pd.read_table(path, header=None, engine='python', sep=',')
    rename_area = dict()
    start_pos = data.iloc[:, 1].values
    end_pos = data.iloc[:, 2].values
    data = data.values

    num = 0
    for i in range(len(start_pos)):
        if not rename_area.__contains__(start_pos[i]):  # 如果已经重命名的区域没有包含start_pos[i]
            num += 1
            rename_area.setdefault(start_pos[i], []).append(num)  # 则添加start_pos[i]
        if not rename_area.__contains__((end_pos[i])):
            num += 1
            rename_area.setdefault(end_pos[i], []).append(num)

    # 输出字典中区域个数
    print("字典内区域有 : {}".format(len(rename_area)))

    for j in range(len(start_pos)):
        start_pos[j] = rename_area.get(start_pos[j])[0]
        end_pos[j] = rename_area.get(end_pos[j])[0]

    time = data[:, 0]
    for t in range(len(time)):
        if time[t] == 0:
            time[t] = 24
    # 打包新数据
    new_data = [list(item) for item in zip(time, start_pos, end_pos, data[:, 3])]
    new_data = np.array(new_data)
    print(new_data.shape)
    # 将数据导入new_transition_probability
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    np.savetxt("../dataset/new_area_transition_probability", new_data, fmt='%.00f,%.00f,%.00f,%.020f', delimiter=',')
    print("完成区域重命名")
    return


if __name__ == '__main__':
    # new_transition_probability()
    lstm_transition_data()
