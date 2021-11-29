# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 07:44:09 2021

@author: kkrao
random forest regression of pws
"""
import os

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import matplotlib.patches
import sklearn.inspection
import sklearn.metrics

import dirs

sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")

def cleanup_data(path):
    """
    path is where h5 file is stored
    """
    
    store = pd.HDFStore(os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate.h5'))
    df =  store['df']   # save it
    store.close()
    df.drop(["pft","isohydricity",'root_depth', 'hft', 'p50', 'gpmax', 'c', 'g1'],axis = 1, inplace = True)
    df.dropna(inplace = True)
    df.reset_index(inplace = True, drop = True)
    
    return df





def regress(df)
    X = df.drop("pws",axis = 1)
    y = df['pws']
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.33, random_state=32)
    
    # for leaves in [6,7,8,9,10,12, 14, 15]:
        # for decrease in [1e-5, 5e-6,1e-6, 1e-7]:
    leaves = 6
    decrease = 1e-6
    regr = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                      min_impurity_decrease=decrease, n_estimators = 50)
    regr.fit(X_train, y_train)
    score = regr.score(X_test,y_test)
    print(f"[INFO] score={score:0.3f}, leaves={leaves}, decrease={decrease}")
    
        # print(regr.score(X_test,y_test))
    # result = sklearn.inspection.permutation_importance(regr, X_test, y_test, 
                                    # n_repeats=10, random_state=0)
    
    # heights = result.importances_mean
    heights = regr.feature_importances_
    ticks = X.columns
    
    # hft_inds = [i for i in range(len(ticks)) if "hft" in ticks[i]]
    # hft_all_imp = heights[hft_inds].sum()
    # heights = np.delete(heights, hft_inds)
    # ticks = [tick for tick in ticks if "hft" not in tick]
    # ticks = ticks + ["hft"]
    # heights = np.append(heights, hft_all_imp)
    
    # order = np.argsort(heights)
    
    # heights = heights[order]
    # ticks = np.array(ticks)[order]
    
    # enc = sklearn.preprocessing.OneHotEncoder()
    # df = df.join(pd.DataFrame(enc.fit(df.hft.values.reshape(-1,1)).\
                                      # transform(df.hft.values.reshape(-1,1)).toarray(),\
                              # columns = [f"hft{i}" for i in range(1,7)]))
    # df = df.drop(["hft"], axis =1)
    
    #convert Ks to log
    # df.ks = np.log10(df.ks)
    # df.drop(['elevation', 'aspect', 'slope', 'twi','dry_season_length', 'ndvi', \
             # 'vpd_mean', 'vpd_std'], axis = 1, inplace = True)
    # print(df.shape)
    
    green = "yellowgreen"
    brown = "saddlebrown"
    blue = "dodgerblue"
    yellow = "khaki"
    
    imp = pd.DataFrame(index = ticks, columns = ["importance"], data = heights)
    plant = ['canopy_height', "agb",'ndvi']
    soil = ['sand',  'clay', 'silt','thetas', 'ks']
    climate = [ 'dry_season_length', 'vpd_mean', 'vpd_std']
    topo = ['elevation', 'aspect', 'slope', 'twi',"dist_to_water"]
    
    def colorize(x):
        if x in plant:
            return green
        elif x in soil:
            return brown
        elif x in climate:
            return blue
        else:
            return yellow
    imp["color"] = imp.index
    imp.color = imp.color.apply(colorize)
    imp["symbol"] = imp.index
    # imp.loc["hft","symbol"] = "Hydraulic\nfunctional type"
    # imp.loc["pft","symbol"] = "PFT"
    imp.loc["sand","symbol"] = "Sand fraction"
    imp.loc["clay","symbol"] = "Clay fraction"
    imp.loc["silt","symbol"] = "Silt fraction"
    imp.loc["canopy_height","symbol"] = "Canopy height"
    imp.loc["ks","symbol"] = "K$_s$"
    imp.loc["thetas","symbol"] = r"Porosity"
    # imp.loc["root_depth","symbol"] = "Root depth"
    # imp.loc["g1","symbol"] = "$g_1$"
    # imp.loc["gpmax","symbol"] = "Max. xylem\nconductance"
    # imp.loc["isohydricity","symbol"] = "Isohydricity"
    # imp.loc["c","symbol"] = "Xylem\ncapacitance"
    # imp.loc["p50","symbol"] = "$\psi_{50}$"
    imp.sort_values("importance", ascending = True, inplace = True)
    print(imp.groupby("color").sum().round(2))

return regr, imp

def plot_importance()
    fig, ax = plt.subplots(figsize = (3.5,6))
    imp.plot.barh(y = "importance",x="symbol",color = imp.color, edgecolor = "grey", ax = ax)

    legend_elements = [matplotlib.patches.Patch(facecolor=green, edgecolor='grey',
                             label='Plant'), 
                       matplotlib.patches.Patch(facecolor=brown, edgecolor='grey',
                             label='Soil'), 
                       matplotlib.patches.Patch(facecolor=yellow, edgecolor='grey',
                             label='Topography'), 
                       matplotlib.patches.Patch(facecolor=blue, edgecolor='grey',
                             label='Climate')]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Variable importance")
    ax.set_ylabel("")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    

def plot_preds_actual(X_test, y_test)

    y_hat =regr.predict(X_test)
    
    fig, ax = plt.subplots(figsize = (3,3))
    ax.scatter(y_hat, y_test, s = 1, alpha = 0.05, color = "k")
    ax.set_xlabel("Predicted PWS")
    ax.set_ylabel("Actual PWS")
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.annotate(f"R$^2$={score:0.2f}", (0.1,0.9),xycoords = "axes fraction", ha = "left")

def plot_importance_by_category()

    combined = pd.DataFrame({"category":["plant","climate","soil","topography"], "color":[green, blue, brown, yellow]})
    combined = combined.merge(imp.groupby("color").sum(), on = "color")
    
    combined = combined.sort_values("importance")
    fig, ax = plt.subplots(figsize = (3.5,2))
    combined.plot.barh(y = "importance",x="category",color = combined.color, edgecolor = "grey", ax = ax,legend =False )
    # ax.set_yticks(range(len(ticks)))
    # ax.set_yticklabels(ticks)
    
    
    ax.set_xlabel("Variable importance")
    ax.set_ylabel("")
    # ax.set_title("Hydraulic traits' predictive\npower for PWS", weight = "bold")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()


## PDPs

def plot_pdp()
    features = [3,10,6,4,12,11,15]
    sklearn.inspection.PartialDependenceDisplay.from_estimator(regr, X_test, features)

def main():
    #%% Load data
    path = os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate.h5')
    df = cleanup_data(path)
    
    
    
    
if __name__ == "__main__":
    main()

