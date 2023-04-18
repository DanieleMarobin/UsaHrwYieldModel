"""
This file relies on the library 'pygad' for the Genetic Algorithms calculations
Unfortunately there are certain functions that do not accept external inputs
so the only way to pass variables to them is to have some global variables
"""

import sys

import re
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
import subprocess
from datetime import datetime as dt
from datetime import timedelta
from copy import deepcopy
import concurrent.futures

import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import pygad

import QuickStats as qs

import SnD as us
import Weather as uw
import Modeling as um
import Charts as uc
import GLOBAL as GV

import warnings; warnings.filterwarnings("ignore")


# Model Core
def initialize_global_variables(y='Yield', n_var=7, save_name='GA_HRW'):
    '''
        This initialization is what drives the whole model searching process.

        The inputs are 3 and very simple:
            - y         : which variable I need to model (ex: 'CORN - ACRES PLANTED|NOT SPECIFIED|YEAR')
            - n_var     : how many variables I want to use to model the above 'y' (ex: 5)
            - save_name : name to save te results (ex: GA_Prospective_Planting_5)

        y = 'Yield'
    '''
    # Core Variables

    global ref_year
    global ref_year_start

    global multi_ww_dt_s
    global multi_ww_dt_e
    global min_single_window

    global multi_ww_freq_start
    global multi_ww_freq_end
    global zeros_cols_limit
    
    # 350'000+ columns
    if False:
        # I didn't write in one line because it is easier to copy this part in Jupter (to test)
        ref_year = GV.CUR_YEAR # Used for creating both 'train' and 'predict' df, so important to have it right    
        ref_year_start=dt(ref_year-1,7,1)

        multi_ww_dt_s=dt(ref_year-1,8,15)
        multi_ww_dt_e=dt(ref_year  ,6,30)
        min_single_window = 7 # in days, to avoid having 1 day window variables

        multi_ww_freq_start='1D'
        multi_ww_freq_end='1D'

        zeros_cols_limit = 0.9 # 0.9 means: remove columns that are 0 more than 90% of the time


    # trying to speed it up by reducing the number of columns
    if True:
        # I didn't write in one line because it is easier to copy this part in Jupter (to test)
        ref_year = GV.CUR_YEAR # Used for creating both 'train' and 'predict' df, so important to have it right    
        ref_year_start=dt(ref_year-1,7,1)

        multi_ww_dt_s=dt(ref_year-1,8,15)
        multi_ww_dt_e=dt(ref_year  ,6,30)
        min_single_window = 7 # in days, to avoid having 1 day window variables

        multi_ww_freq_start='7D'
        multi_ww_freq_end='7D'

        zeros_cols_limit = 0.8 # 0.9 means: remove columns that are 0 more than 90% of the time

    dormancy_s = um.seas_day(dt(ref_year-1,12,1), ref_year_start=ref_year_start)
    dormancy_e = um.seas_day(dt(ref_year  , 3,1), ref_year_start=ref_year_start)

    global dormancy; dormancy = set(np.arange(dormancy_s, dormancy_e, dtype='datetime64[D]'))
    global post_dormancy_prec; post_dormancy_prec = dormancy_e + pd.DateOffset(months=1)
    
    global years; years = list(range(1985,GV.CUR_YEAR+1)) # to get the data    
    
    global min_coverage; min_coverage = 90.0 # 60 # in days
    
    global min_train_size; min_train_size = 10

    global folder; folder= r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\Data\Models\USA HRW Yield\\'

    # global X_cols_fixed; X_cols_fixed = ['year', 'USA_Fdd-5_Aug15-Jun24']
    global X_cols_fixed; X_cols_fixed = ['year']
    global X_cols_excluded; X_cols_excluded = [] # As obviously I cannot use current information to predict current variables
        
    # Genetic Algorithm
    global initial_p_values_threshold; initial_p_values_threshold=0.5
    global initial_corr_threshold; initial_corr_threshold=1.0
    
    global final_p_values_threshold; final_p_values_threshold=0.05
    global final_corr_threshold; final_corr_threshold=0.85
                                    
    global num_generations; num_generations = 10000000000

    global solutions_per_population; solutions_per_population = 10 # Number of solutions (i.e. chromosomes) within the population
    global num_parents_mating; num_parents_mating = 4

    global parent_selection_type; parent_selection_type='rank'    
    global mutation_type; mutation_type='random'
    global mutation_probability; mutation_probability=1.0

    global stop_criteria; stop_criteria=["reach_1000000", "saturate_20000"]

    # Secondary
    global y_col; y_col = y
    global GA_pref; GA_pref={}
    global GA_n_variables; GA_n_variables = n_var
    global save_file; save_file= folder + save_name+'_'+str(n_var)

    return True

def Define_Scope():
    """
    'geo_df':
        it is a dataframe (selection of rows of the weather selection file)
    'geo_input_file': 
        it needs to match the way files were named by the API
            GV.WS_STATE_NAME    ->  Mato Grosso_Prec.csv
            GV.WS_STATE_ALPHA   ->  MT_Prec.csv
            GV.WS_STATE_CODE    ->  51_Prec.csv

    'geo_output_column':
        this is how the columns will be renamed after reading the above files (important when matching weight matrices, etc)
            GV.WS_STATE_NAME    ->  Mato Grosso_Prec
            GV.WS_STATE_ALPHA   ->  MT_Prec
            GV.WS_STATE_CODE    ->  51_Prec
    """

    fo={}

    # Geography (Read the comment above, Expand the section if it is hidden/collapsed)
    geo = uw.get_w_sel_df()
    fo['geo_df'] = geo[geo[GV.WS_COUNTRY_ALPHA] == 'USA']
    fo['geo_input_file'] = GV.WS_UNIT_ALPHA 
    fo['geo_output_column'] = GV.WS_UNIT_ALPHA

    # Weather Variables
    fo['w_vars'] = [
        GV.WV_PREC, 
        GV.WV_TEMP_MAX, 
        GV.WV_TEMP_MIN, 

        GV.WV_SDD+'10',
        GV.WV_SDD+'11',
        GV.WV_SDD+'12', 
        GV.WV_SDD+'15', 
        GV.WV_SDD+'17',
        GV.WV_SDD+'18',
        GV.WV_SDD+'19', 
        GV.WV_SDD+'20',
        GV.WV_SDD+'21', 
        GV.WV_SDD+'22', 
        GV.WV_SDD+'23', 
        GV.WV_SDD+'24', 
        GV.WV_SDD+'25',
        GV.WV_SDD+'27',
        GV.WV_SDD+'30', 

        GV.WV_FDD+'0', 
        GV.WV_FDD+'-1',
        GV.WV_FDD+'-2', 
        GV.WV_FDD+'-3',
        GV.WV_FDD+'-4',
        GV.WV_FDD+'-5', 
        GV.WV_FDD+'-10', 
        ]

    # Time
    # the below try it is only for when I test in the Jupiter Notebook (because it doesn't run the global initialization)
    try:
        fo['years']=years
    except:
        fo['years']=list(range(1985,GV.CUR_YEAR+1))

    return fo

def Get_Data_Single(scope: dict, var: str = 'yield', fo = {}):
    commodity = 'WHEAT, WINTER, RED, HARD'
    if (var=='yield'):
        df = qs.get_USA_yields(commodity=commodity, years=scope['years'])
        df['year']=df.index
        df = df[['year','Value']]
        df = df.rename(columns={'Value':'Yield'})
        return df

    elif (var=='weights'):
        return us.get_USA_prod_weights(commodity=commodity,aggregate_level= 'STATE', years= scope['years'], pivot_column='state_alpha')

    elif (var=='w_df_all'):
        return uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])

    elif (var=='w_w_df_all'):
        return uw.weighted_w_df_all(fo['w_df_all'], fo['weights'], output_column='USA')

    return fo

def Get_Data_All_Parallel(scope):
    # https://towardsdatascience.com/multi-tasking-in-python-speed-up-your-program-10x-by-executing-things-simultaneously-4b4fc7ee71e

    fo={}

    # Time
    fo['years']=scope['years']

    # Space
    fo['locations']=scope['geo_df'][GV.WS_STATE_ALPHA]

    download_list=['yield','weights', 'w_df_all'] # 'planting_progress','blooming_progress',
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for variable in download_list:
            results[variable] = executor.submit(Get_Data_Single, scope, variable, fo)
    
    for var, res in results.items():
        fo[var]=res.result()
    
    # Weighted Weather: it is here because it needs to wait for the 2 main in ingredients (1) fo['w_df_all'], (2) fo['weights'] to be calculated first
    variable = 'w_w_df_all'
    fo[variable]  = Get_Data_Single(scope, variable, fo)

    return fo

def from_raw_data_to_model_df(raw_data):
    dfs=[]
    # add the yield (as it is the 'Y' variable)
    yield_df = raw_data['yield']
    dfs.append(yield_df)

    # add add the weather windows
    ww_df = um.generate_weather_windows_df(raw_data['w_w_df_all']['hist'], date_start=multi_ww_dt_s, date_end=multi_ww_dt_e, ref_year_start=ref_year_start, freq_start=multi_ww_freq_start, freq_end=multi_ww_freq_end, min_single_window=min_single_window)
    print('ww_df.shape (All):', ww_df.shape)
    
    ww_df = select_weather_windows(ww_df, ref_year_start, zeros_cols_limit)
    print('ww_df.shape (Selected):', ww_df.shape)

    dfs.append(ww_df)

    # Concatenating all the info
    df_model = pd.concat(dfs, sort=True, axis=1, join='inner')

    return df_model

def select_weather_windows(ww_df, ref_year_start, zeros_cols_limit):
    # This excludes all the columns whose value is 0 more than 90% of the time
    mask = ((ww_df == 0).sum(axis=0) / len(ww_df)) >= zeros_cols_limit
    ww_df=ww_df.loc[:, ~mask]

    # In case I needed it for later
    if False:
        sel_col=ww_df.columns[0]
        wws = um.var_windows_from_cols([sel_col], ref_year_start=ref_year_start)

        v=wws[0]['variables'][0] # variable
        s=wws[0]['windows'][0]['start'] # start
        e=wws[0]['windows'][0]['end'] # end

        print(v,s,e)

    return ww_df

def Build_DF(raw_data, instructions, saved_m):
    """
    The model DataFrame has 11 Columns:
            1) Yield (y)
            8) Variables
            1) Constant (added to be able to fit the model with 'statsmodels.api')

            1+8+1 = 10 Columns
    """

    w_all=instructions['WD_All'] # 'simple'->'w_df_all', 'weighted'->'w_w_df_all'
    WD=instructions['WD']
    w_df = raw_data[w_all][WD]
    
    ref_year = instructions['ref_year']
    ref_year_start = instructions['ref_year_start']
    
    wws = um.var_windows_from_cols(saved_m.params.index, ref_year_start=ref_year_start)
    df = um.extract_yearly_ww_variables(w_df = w_df, var_windows= wws, ref_year=ref_year, ref_year_start=ref_year_start)

    df = pd.concat([raw_data['yield'], df], sort=True, axis=1, join='outer')
    df['year']=df.index
    df = sm.add_constant(df, has_constant='add')

    return df
    
def Build_Pred_DF(raw_data, instructions, year_to_ext = GV.CUR_YEAR,  date_start=dt.today(), date_end=None, trend_yield_case= False, saved_m=None):
    """
    for predictions I need to:
        1) extend the variables:
                1.1) Weather
                1.2) All the Milestones
                1.3) Recalculate the Intervals (as a consequence of the Milestones shifting)

        2) cut the all the rows before CUR_YEAR so that the calculation is fast:
             because I will need to extend every day and recalculate
    """
    
    dfs = []
    w_all=instructions['WD_All']
    WD=instructions['WD']
    ext_dict = instructions['ext_mode']
    ref_year=instructions['ref_year']
    ref_year_start=instructions['ref_year_start']

    raw_data_pred = deepcopy(raw_data)
    w_df = raw_data[w_all][WD]
    
    if (date_end==None): date_end = w_df.index[-1] # this one to check well what to do
    days_pred = list(pd.date_range(date_start, date_end))

    for i, day in enumerate(days_pred):
        if trend_yield_case:
            keep_duplicates='last'
        else:
            keep_duplicates='first'

        # Extending the Weather        
        if (i==0):
            # Picks the extension column and then just uses it till the end            
            raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df[w_df.index<=day], return_dict_col_seas=True, var_mode_dict=ext_dict, ref_year=ref_year, ref_year_start=ref_year_start,keep_duplicates= keep_duplicates)
        else:
            raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df[w_df.index<=day], input_dict_col_seas = dict_col_seas, var_mode_dict=ext_dict, ref_year=ref_year, ref_year_start=ref_year_start,keep_duplicates=keep_duplicates)
        
        # Build the 'Simulation' DF    
        w_df_pred = Build_DF(raw_data_pred, instructions, saved_m) # Take only the GV.CUR_YEAR row and append

        # print(w_df_pred.shape)        
        # print(w_df_pred)
        # if i==5:
        #     return dfs

        # Append row to the final matrix (to pass all at once for the daily predictions)
        dfs.append(w_df_pred.loc[year_to_ext:year_to_ext]) # right
    
    fo = pd.concat(dfs)

    # This one is to be able to have the have the 'full_analysis' chart
    # and also it makes a lot of sense:
    #       - the fo shows for each day, which dataset should be used
    fo.index= days_pred.copy()
    return fo