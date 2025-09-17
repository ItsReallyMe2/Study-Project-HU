import pandas as pd
from pprint import pprint as pp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from mpl_toolkits.mplot3d import Axes3D
#from color_map import custom_cmap_func
from sklearn.linear_model import LinearRegression

'''functions for visualization'''

'''Not used in this analysis'''
def variance_components(pca, data):
    X_pca = pca.fit_transform(data)
    plt.bar(range(1,len(pca.explained_variance_)+1), pca.explained_variance_)
    plt.xticks(np.arange(0,len(pca.explained_variance_)+1, 1), rotation=90)
    plt.ylabel('Explained variance')
    plt.xlabel('Number of principal components')
    plt.plot(range(1, len(pca.explained_variance_)+1), 
         np.cumsum(pca.explained_variance_),
         c='red', 
         label='Cumulative Explained Variance')
    plt.legend(loc='upper left')
    plt.show()
    return None 
def variance_ratio(data, n=None):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(data)
    x = np.array(range(1, len(pca.explained_variance_ratio_)+1))
    y = pca.explained_variance_ratio_
    plt.xticks(rotation=90)
    plt.bar(x, pca.explained_variance_ratio_, color='#a6cee3', edgecolor='black')
    plt.xticks(np.arange(0,len(pca.explained_variance_ratio_)+1, 1))
    plt.xlim(0, len(pca.explained_variance_ratio_)+1)
    plt.ylim(0, pca.explained_variance_ratio_.max()+0.10)
    plt.xlabel('Number of principal components')
    plt.ylabel('Explained variance ratio')
    plt.show()
    return None
def sparse_principal_component_2d_timesample(data, save, n=2, x=1, y=2, c_map='tab20'):
    pca = SparsePCA(n_components=n, n_jobs=4)
    X_pca = pca.fit_transform(data)
    if save == True:
        vectors_pc(pca, data, 1, 1_0)
        vectors_pc(pca, data,2,2_0)
    plt.figure(figsize=(8, 7))
    plt.xlabel('PC'+str(x),fontweight='bold')
    plt.ylabel('PC'+str(y), fontweight='bold')
    plt.title('PCA of Timeseries as Samples', fontweight='bold')
    scatter = plt.scatter(X_pca[:,x-1], X_pca[:,y-1], c=data.index.astype(float), cmap=c_map, alpha=0.8)
    plt.legend(
        scatter.legend_elements(prop='colors', num=None)[0],
        data.index.unique(),
        loc="center left",
        title="Time in h",
        bbox_to_anchor=(1, 0.5),)
    plt.tight_layout()
    plt.axis('equal')
    plt.show() 
    return None
def principal_component_2d_timesample_to_save(data, fig, x=1, y=2, pca_f=None, c_map='tab20'):
    pca = PCA(pca_f)
    X_pca = pca.fit_transform(data)
    plt.figure(figsize=(8, 7))
    plt.xlabel('PC'+str(x),fontweight='bold')
    plt.ylabel('PC'+str(y), fontweight='bold')
    plt.title('PCA of Timeseries as Samples', fontweight='bold')
    scatter = plt.scatter(X_pca[:,x-1], X_pca[:,y-1], c=data.index.astype(float), cmap=c_map, alpha=0.8)
    plt.legend(
        scatter.legend_elements(prop='colors', num=None)[0],
        data.index.unique(),
        loc="center left",
        title="Time in h",
        bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig('/Users/maksimsgolubovics/Python_VScode/Studienprojekt/PCA_by_study/fig_'+str(fig), dpi=300)
    return None
def principal_component_2d_genesample(data, x=1, y=2, pca_f=None):
    pca = PCA(pca_f)
    X_pca = pca.fit_transform(data)
    plt.figure(figsize=(8, 7), dpi=100)
    plt.xlabel('PC'+str(x),fontweight='bold')
    plt.ylabel('PC'+str(y), fontweight='bold')
    plt.title('PCA of Gene Expression as Sample', fontweight='bold')
    scatter = plt.scatter(X_pca[:,x-1], X_pca[:,y-1], alpha=0.8)
    plt.show() 
    return None
def vectors_pc(data, x):
    pca = PCA()
    pca.fit(data)
    loadings = pca.components_[x-1]
    feature_names = data.columns
    pc_importans = pd.Series(loadings, index=feature_names).sort_values(ascending=False)
    pp(pc_importans)
    return None

'''Often used in analysis'''
def pairplot_psa_6(data, palette, hue='time'):
    pca = PCA(n_components=6)
    pca_t = pca.fit_transform(data)
    pca_t_df = pd.DataFrame(index=data.index, columns=['PC1','PC2','PC3','PC4','PC5','PC6'], data=pca_t).reset_index()
    sns.pairplot(pca_t_df, hue=hue, palette=palette, plot_kws={'alpha':0.8, 's':20})
    plt.show()
    return None
def pairplot_psa_3(data, palette, hue='time'):
    pca = PCA(n_components=3)
    pca_t = pca.fit_transform(data)
    pca_t_df = pd.DataFrame(index=data.index, columns=['PC1','PC2','PC3'], data=pca_t).reset_index()
    sns.pairplot(pca_t_df, hue=hue, palette=palette, plot_kws={'alpha':0.8, 's':20})
    plt.show()
    return None
def principal_component_2d_timesample(data, label, x=1, y=2, c_map='tab20', remove_legend=False):
    pca = PCA()
    X_pca = pca.fit_transform(data)
    plt.figure(figsize=(8, 7))
    plt.xlabel('PC'+str(x),fontweight='bold')
    plt.ylabel('PC'+str(y), fontweight='bold')
    plt.title('PCA of '+label+' as Samples', fontweight='bold')
    scatter = plt.scatter(X_pca[:,x-1], X_pca[:,y-1], c=pd.Series(data.index).astype('category').cat.codes, cmap=c_map, 
                          alpha=0.8, s=30, edgecolors='black', linewidths=0.15)
    legend = plt.legend(
        scatter.legend_elements(prop='colors', num=None)[0],
        pd.Series(data.index).astype('category').cat.categories,
        loc="center left",
        title=label,
        bbox_to_anchor=(1, 0.5),)
    plt.tight_layout()
    plt.axis('equal')
    if remove_legend is True:
        legend.remove()
    plt.show() 
    return None
def principal_component_3d_timesample(data, label, x=1, y=2, z=3, pca_f=None, c_map='tab20'):
    pca = PCA(pca_f)
    X_pca = pca.fit_transform(data)
    
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,x-1],X_pca[:,y-1],X_pca[:,z-1], c=pd.Series(data.index).astype('category').cat.codes, cmap=c_map,
                            s=30, edgecolors='black', linewidths=0.15)
    ax.set_xlabel('PC'+str(x),fontweight='bold')
    ax.set_ylabel('PC'+str(y),fontweight='bold')
    ax.set_zlabel('PC'+str(z),fontweight='bold')

    plt.legend(
        scatter.legend_elements(prop='colors', num=None)[0],
        pd.Series(data.index).astype('category').cat.categories,
        loc="center left",
        title=label,
        bbox_to_anchor=(1, 0.5),)
    ax.tick_params(labelbottom=False, labelleft=False)

    plt.show()
    return None
def variance_ratio(data, n=None):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(data)
    x = np.array(range(1, len(pca.explained_variance_ratio_)+1))
    y = pca.explained_variance_ratio_
    plt.xticks(rotation=90)
    plt.bar(x, pca.explained_variance_ratio_, color='#a6cee3', edgecolor='black')
    plt.xticks(np.arange(0,len(pca.explained_variance_ratio_)+1, 1))
    plt.xlim(0, len(pca.explained_variance_ratio_)+1)
    plt.ylim(0, pca.explained_variance_ratio_.max()+0.10)
    plt.xlabel('Number of principal components')
    plt.ylabel('Explained variance ratio')
    plt.show()
    return None
def residual(data, columns=['time']):
    data_dummy =pd.get_dummies(data.reset_index(), columns=columns, dtype=int)
    x = -len(data.reset_index().set_index(columns).index.unique())
    data_dummy_only = data_dummy.iloc[:, x:]
    model = LinearRegression(fit_intercept=True)
    fit = model.fit(X=data_dummy_only,y=data)
    prediction = model.predict(data_dummy_only)
    residual = (data - prediction)
    return residual
def residual_dummy(data, data_dummy_1 ,columns):
    data_dummy =pd.get_dummies(data_dummy_1, columns=[columns], dtype=int)
    model = LinearRegression(fit_intercept=True)
    fit = model.fit(X=data_dummy,y=data)
    prediction = model.predict(data_dummy)
    residual = (data - prediction)
    return residual
def visualization_of_dec_tools_3d(dec, data, label, x=1, y=2, z=3, c_map='tab20'):
    dec = dec
    X_dec = dec.fit_transform(data)
    
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_dec[:,x-1],X_dec[:,y-1],X_dec[:,z-1], c=pd.Series(data.index).astype('category').cat.codes, cmap=c_map,
                            s=30, edgecolors='black', linewidths=0.15)
    ax.set_xlabel(str(x),fontweight='bold')
    ax.set_ylabel(str(y),fontweight='bold')
    ax.set_zlabel(str(z),fontweight='bold')

    plt.legend(
        scatter.legend_elements(prop='colors', num=None)[0],
        pd.Series(data.index).astype('category').cat.categories,
        loc="center left",
        title=label,
        bbox_to_anchor=(1, 0.5),)
    ax.tick_params(labelbottom=False, labelleft=False)

    plt.show()
    return None
def visualization_of_dec_tools_2d(dec, data, label, x=1, y=2, c_map='tab20'):
    dec = dec
    X_dec = dec.fit_transform(data)
    plt.figure(figsize=(8, 7))
    plt.xlabel(str(x),fontweight='bold')
    plt.ylabel(str(y), fontweight='bold')
    scatter = plt.scatter(X_dec[:,x-1], X_dec[:,y-1], c=pd.Series(data.index).astype('category').cat.codes, cmap=c_map, 
                          alpha=0.8, s=30, edgecolors='black', linewidths=0.15)
    plt.legend(
        scatter.legend_elements(prop='colors', num=None)[0],
        pd.Series(data.index).astype('category').cat.categories,
        loc="center left",
        title=label,
        bbox_to_anchor=(1, 0.5),)
    plt.tight_layout()
    plt.axis('equal')
    plt.show() 
    return None

