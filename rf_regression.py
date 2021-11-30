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


def get_categories_and_colors():
    """
    colors and categorize to combine feature importance chart
    """
    
    green = "yellowgreen"
    brown = "saddlebrown"
    blue = "dodgerblue"
    yellow = "khaki"
    
    plant = ['canopy_height', "agb",'ndvi']
    soil = ['sand',  'clay', 'silt','thetas', 'ks']
    climate = [ 'dry_season_length', 'vpd_mean', 'vpd_std']
    topo = ['elevation', 'aspect', 'slope', 'twi',"dist_to_water"]
    
    return green, brown, blue, yellow, plant, soil, climate, topo

def prettify_names(names):
    new_names = {"ks":"K$_s$",
                 "ndvi":"NDVI",
                 "vpd_mean":"VPD$_{mean}$",
                 "thetas":"Soil porosity",
                 "elevation":"Elevation",
                 "dry_season_length":"Dry season length"
                 }
    return [new_names[key] for key in names]
    
    
    
    
    
def regress(df):
    """
    Regress features on PWS using rf model
    Parameters
    ----------
    df : columns should have pws and features

    Returns:
        X_test:dataframe of test set features
        y_test: Series of test set pws
        regr: trained rf model (sklearn)
        imp: dataframe of feature importance in descending order
    -------
    

    """
    # separate data into features and labels
    X = df.drop("pws",axis = 1)
    y = df['pws']
    # separate into train and test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.33, random_state=32)
    # Checking if leaves or node_impurity affects performance
    # after running found that it has almost no effect (R2 varies by 0.01)
    # for leaves in [6,7,8,9,10,12, 14, 15]:
        # for decrease in [1e-5, 5e-6,1e-6, 1e-7]:
    # choose min leaves in terminal node and node impurity
    leaves = 6
    decrease = 1e-6
    # construct rf model
    regr = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                      min_impurity_decrease=decrease, n_estimators = 50)
    # train
    regr.fit(X_train, y_train)
    # test set performance
    score = regr.score(X_test,y_test)
    print(f"[INFO] score={score:0.3f}, leaves={leaves}, decrease={decrease}")
    
    # assemble all importance with feature names and colors
    heights = regr.feature_importances_
    ticks = X.columns
  
    green, brown, blue, yellow, plant, soil, climate, topo = get_categories_and_colors()
    
    imp = pd.DataFrame(index = ticks, columns = ["importance"], data = heights)
    
    def _colorize(x):
        if x in plant:
            return green
        elif x in soil:
            return brown
        elif x in climate:
            return blue
        else:
            return yellow
    imp["color"] = imp.index
    imp.color = imp.color.apply(_colorize)
    imp["symbol"] = imp.index
    # cleanup variable names
    imp.loc["sand","symbol"] = "Sand fraction"
    imp.loc["clay","symbol"] = "Clay fraction"
    imp.loc["silt","symbol"] = "Silt fraction"
    imp.loc["canopy_height","symbol"] = "Canopy height"
    imp.loc["ks","symbol"] = "K$_s$"
    imp.loc["thetas","symbol"] = r"Porosity"
    imp.sort_values("importance", ascending = True, inplace = True)
    print(imp.groupby("color").sum().round(2))

    return X_test, y_test, regr, score, imp

    

def plot_preds_actual(X_test, y_test, regr, score):
    """
    Plot of predictions vs actual data
    """
    y_hat =regr.predict(X_test)
    
    fig, ax = plt.subplots(figsize = (3,3))
    ax.scatter(y_hat, y_test, s = 1, alpha = 0.05, color = "k")
    ax.set_xlabel("Predicted PWS")
    ax.set_ylabel("Actual PWS")
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.annotate(f"R$^2$={score:0.2f}", (0.1,0.9),xycoords = "axes fraction", ha = "left")
    return ax


def plot_importance(imp):
    """
    plot feature importance for all features

    Parameters
    ----------
    imp : dataframe returned by regress

    Returns
    -------
    ax: axis handle

    """
    
    fig, ax = plt.subplots(figsize = (3.5,6))
    green, brown, blue, yellow, plant, soil, climate, topo = get_categories_and_colors()

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
    return ax

def plot_importance_by_category(imp):
    """
    Feature importance combined by categories
    """
    green, brown, blue, yellow, plant, soil, climate, topo = get_categories_and_colors()
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

def plot_pdp(regr, X_test):
    """
    Partial dependance plot
    requires scikit-learn>=0.24.2
    Parameters
    ----------
    regr : trained rf regression
    X_test : test set data for creating plot
    """
    
    features = [3,10,6,4,12,11]
    feature_names = list(X_test.columns[features])
    feature_names = prettify_names(feature_names)
    for feature, feature_name in zip(features, feature_names):
        pd_results = sklearn.inspection.partial_dependence(regr, X_test, feature)
        fig, ax = plt.subplots(figsize = (4,4))
        ax.plot(pd_results[1][0], pd_results[0][0])
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Plant-water sensitivity")
        plt.show()

    
def main():
    #%% Load data
    path = os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate.h5')
    df = cleanup_data(path)
    #%% train rf
    X_test, y_test, regr, score,  imp = regress(df)
    #%% make plots
    ax = plot_importance(imp)
    ax = plot_importance_by_category(imp)
    ax = plot_preds_actual(X_test, y_test, regr, score)
    plot_pdp(regr, X_test)
    

if __name__ == "__main__":
    main()

