from matplotlib.colors import LinearSegmentedColormap

'''Custom_cmap'''
c = [
    '#a6cee3','#77b3a9','#1f78b4','#ffffb3','#b2df8a','#bebada',
    '#33a02c','#fb8072','#fb9a99','#80b1d3','#e31a1c','#fdb462',
    '#d9a462','#b3de69','#ff7f00','#fccde5','#cab2d6','#d9d9d9',
    '#6a3d9a','#bc80bd','#ffff99','#ccebc5','#b15928','#ffed6f']
c_34 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#17becf", "#393b79", "#637939", "#8c6d31",
    "#843c39", "#7b4173", "#6b6ecf", "#31a354", "#756bb1", "#9e9ac8",
    "#5254a3", "#3182bd", "#e6550d", "#a63603", "#4daf4a", "#984ea3",
    "#ff33cc", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6",
    "#6a3d9a", "#b15928", "#199d76", "#d95f02"
]
c_4r_0_irl = ['orange' if _ in [0,5,11,16,21,27] else '#a6cee3' for _ in range(32)]
c_4r_1_irl = ['orange' if _ in [1,7,12,17,23,28] else '#a6cee3' for _ in range(32)]
c_4r_2_irl = ['orange' if _ in [3,8,13,19,24,29] else '#a6cee3' for _ in range(32)]
c_4r_3_irl = ['orange' if _ in [4,9,15,20,25,31] else '#a6cee3' for _ in range(32)]

def custom_cmap_func(var_name='c'):
    colors = globals()[var_name]
    custom_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    return custom_cmap