 # -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:46:32 2023

@author: Freddy J. Orozco R.
"""

import streamlit as st
import hydralit_components as hc
import datetime
import base64
import pandas as pd
from io import BytesIO
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as mplt
import matplotlib.font_manager as font_manager
import mplsoccer
from mplsoccer import Pitch, VerticalPitch, FontManager
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
import seaborn as sns
from matplotlib import colors as mcolors
import requests
from PIL import Image
from matplotlib.patches import Rectangle
from streamlit_option_menu import option_menu
from datetime import datetime
from radar_chart2 import Radar

#make it look nice from the start
st.set_page_config(layout='wide', page_title='Future Talents VII - Dashboard', initial_sidebar_state='collapsed')

#####################################################################################################################################################

font_path = 'Resources/keymer-bold.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop2 = font_manager.FontProperties(fname=font_path)

font_path2 = 'Resources/BasierCircle-Italic.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path2)
prop3 = font_manager.FontProperties(fname=font_path2)

#####################################################################################################################################################

hex_list2 = ['#121214', '#D81149', '#FF0050']
#hex_list = ['#121214', '#112F66', '#004DDD']B91845
hex_list4 = ['#5A9212', '#70BD0C', '#83E604']
#hex_list1 = ['#121214', '#854600', '#C36700']
hex_list = ['#121214', '#545454', '#9F9F9F']
hex_list1 = ['#121214', '#695E00', '#C7B200']
#hex_list2 = ['#121214', '#112F66', '#004DDD']
#hex_list = ['#121214', '#11834C', '#00D570']
cmap = sns.cubehelix_palette(start=.25, rot=-.3, light=1, reverse=True, as_cmap=True)
cmap2 = sns.diverging_palette(250, 344, as_cmap=True, center="dark")
cmap3 = sns.color_palette("dark:#FF0046", as_cmap=True)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
 
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def colorlist(color1, color2, num):
    """Generate list of num colors blending from color1 to color2"""
    result = [np.array(color1), np.array(color2)]
    while len(result) < num:
        temp = [result[0]]
        for i in range(len(result)-1):
            temp.append(np.sqrt((result[i]**2+result[i+1]**2)/2))
            temp.append(result[i+1])
        result = temp
    indices = np.linspace(0, len(result)-1, num).round().astype(int)
    return [result[i] for i in indices] 

def soc_pitch_divisions(ax, grids = False):
    '''
    This function returns a vertical football pitch
    divided in specific locations.

    Args:
        ax (obj): a matplotlib axes.
        grids (bool): should we draw the grid lines?
    '''

    # Notice the extra parameters passed to the object
    pitch = VerticalPitch(
        pitch_type = "uefa",
        half = True,
        goal_type='box',
        linewidth = 1,
        pitch_color='None',
        line_color='w',
        line_zorder=0
    )

    pitch.draw(ax = ax)

    # Where we'll draw the lines
    if grids:
        y_lines = [105 - 5.5*x for x in range(1,10)]
        x_lines = [68 - 6.8*x for x in range(1,10)]

        for i in x_lines:
            ax.plot(
                [i, i], [45, 105], 
                color = "#8E8E8E", 
                ls = "--",
                lw = 0.3,
                zorder = -1
            )
        for j in y_lines:
            ax.plot(
                [68, 0], [j, j],
                color = "#8E8E8E", 
                ls = "--",
                lw = 0.3,
                zorder = -1
            )

    return ax
    
#####################################################################################################################################################
#####################################################################################################################################################

# 1. as sidebar menu
with st.sidebar:
    with open("Resources/logobk.png", "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

            st.sidebar.markdown(
                f"""
                <div style="display:table;margin-top:-20%">
                    <img src="data:image/png;base64,{data}" width="200">
                </div>
                """,
                unsafe_allow_html=True,
            )
    selected = option_menu("", ["Rankings", "Player Search", "Comparison"], 
        icons=['trophy', 'search', 'server'], default_index=1, styles={
            "nav-link": {"font-size": "15px"}})
    

# 2. horizontal menu
#selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#    icons=['house', 'cloud-upload', "list-task", 'gear'], 
#    menu_icon="cast", default_index=0, orientation="horizontal")
#selected2

# 3. CSS style definitions
#selected3 = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
#    icons=['house', 'cloud-upload', "list-task", 'gear'], 
#    menu_icon="cast", default_index=0, orientation="horizontal",
#    styles={
#        "container": {"padding": "0!important", "background-color": "#fafafa"},
#        "icon": {"color": "orange", "font-size": "25px"}, 
#        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#        "nav-link-selected": {"background-color": "green"},
#    }
#)

# 4. Manual Item Selection
#if st.session_state.get('switch_button', False):
#    st.session_state['menu_option'] = (st.session_state.get('menu_option',0) + 1) % 4
#    manual_select = st.session_state['menu_option']
#else:
#    manual_select = None
    
#selected4 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#    icons=['house', 'cloud-upload', "list-task", 'gear'], 
#    orientation="horizontal", manual_select=manual_select, key='menu_4')
#st.button(f"Move to Next {st.session_state.get('menu_option',1)}", key='switch_button')
#selected4

# 5. Add on_change callback
#def on_change(key):
#    selection = st.session_state[key]
#    st.write(f"Selection changed to {selection}")
    
#selected5 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'],
#                        icons=['house', 'cloud-upload', "list-task", 'gear'],
#                        on_change=on_change, key='menu_5', orientation="horizontal")
#selected5

###Data
##df = pd.read_excel("MatchesData/matches.xlsx")
df = pd.read_excel("MatchesData/all_matches_prueba2.xlsx")
df['X1'] = (df['X1']*105)/1
df['Y1'] = (df['Y1']*68)/1
df['X2'] = (df['X2']*105)/1
df['Y2'] = (df['Y2']*68)/1
###Convert for left attack-side
for index, row in df.iterrows():
    if row['Atack side'] == 'Left':
        df.at[index, 'X1'] = 105 - row['X1']
        df.at[index, 'Y1'] = 68 - row['Y1']
        df.at[index, 'X2'] = 105 - row['X2']
        df.at[index, 'Y2'] = 68 - row['Y2']

df['Index'] = df['Index'].fillna("")
df['Event'] = df['Action'] + ' - ' + df['Index']
df['PlayerID'] = df['Players']+'-'+df['Team']
event_counts = df.groupby(['PlayerID'])['Event'].agg('count').reset_index()
event_counts2 = df.groupby(['PlayerID'])['Event'].value_counts().unstack(fill_value=0)
dfORIGINAL = df
#PASS FILTER
df = df[(df['Action'] == 'Pass') | (df['Action'] == 'Type pass')].reset_index(drop=True)
df_backup = df
#PROGRESSIVE PASSES
df = df_backup
df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
df['Prog'] = (df['Ending']) / (df['Beginning'])
df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.80 for x in range(len(df.Beginning))]
dfprog = df[df['Progress'] == True].reset_index(drop=True)
dfprogF = dfprog[dfprog['Index'] == 'Miss'].reset_index(drop=True)
dfprogW = dfprog[(dfprog['Index'] == 'Assist') | (dfprog['Index'] == 'Key') | (dfprog['Index'] == 'Second assist') | (dfprog['Index'] == 'Complete')].reset_index(drop=True)
dfprg = dfprog.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfprg.columns = ['PlayerID', 'Team', 'Total Progressive Passes']
dfprg = dfprg.sort_values('Total Progressive Passes', ascending=False)
dfprg0 = dfprg
dfprgs = dfprogW.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfprgs.columns = ['PlayerID', 'Team', 'Successful Progressive Passes']
dfprgs = dfprgs.sort_values('Successful Progressive Passes', ascending=False)
dfprgsF = dfprogF.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfprgsF.columns = ['PlayerID', 'Team', 'Unsuccessful Progressive Passes']
dfprgsF = dfprgsF.sort_values('Unsuccessful Progressive Passes', ascending=False)
dfprgA = dfprg.merge(dfprgs[['PlayerID', 'Successful Progressive Passes']], on='PlayerID', how='outer')
dfprgB = dfprgA.merge(dfprgsF[['PlayerID', 'Unsuccessful Progressive Passes']], on='PlayerID', how='outer')
dfprgB = dfprgB.fillna(0)
dfprgB['% Successful Progressive Passes'] = round((dfprgB['Successful Progressive Passes']*100)/dfprgB['Total Progressive Passes'])
##PASSES TO FINAL THIRD
df = df_backup
df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)
dfpatofithSP = df[(df['Index'] == 'Assist') | (df['Index'] == 'Key') | (df['Index'] == 'Second assist') | (df['Index'] == 'Complete')].reset_index(drop=True)
dfpatofithUP = df[df['Index'] == 'Miss'].reset_index(drop=True)
dfpatofith = df.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpatofith.columns = ['PlayerID', 'Team', 'Total Passes to Final Third']
dfpatofith = dfpatofith.sort_values('Total Passes to Final Third', ascending=False)
dfpatofithW = dfpatofithSP.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpatofithW.columns = ['PlayerID', 'Team', 'Successful Passes to Final Third']
dfpatofithW = dfpatofithW.sort_values(by="Successful Passes to Final Third", ascending=False).reset_index(drop=True)
dfpatofithF = dfpatofithUP.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpatofithF.columns = ['PlayerID', 'Team', 'Unsuccessful Passes to Final Third']
dfpatofithF = dfpatofithF.sort_values(by="Unsuccessful Passes to Final Third", ascending=False).reset_index(drop=True)
dfpatofithA = dfpatofith.merge(dfpatofithW[['PlayerID', 'Successful Passes to Final Third']], on='PlayerID', how='left')
dfpatofithB = dfpatofithA.merge(dfpatofithF[['PlayerID', 'Unsuccessful Passes to Final Third']], on='PlayerID', how='left')
dfpatofithB = dfpatofithB.fillna(0)
dfpatofithB['% Successful Passes to Final Third'] = round((dfpatofithB['Successful Passes to Final Third']*100)/dfpatofithB['Total Passes to Final Third'])
##PASSES TO PENALTY AREA
df = df_backup
# Coordenadas del cuadrilátero
x1_cuadrilatero, y1_cuadrilatero = 88.5, 13.84
x2_cuadrilatero, y2_cuadrilatero = 105, 13.84
x3_cuadrilatero, y3_cuadrilatero = 88.5, 54.16
x4_cuadrilatero, y4_cuadrilatero = 105, 54.16
# Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
condicion1 = (
  (df['X1'] < x1_cuadrilatero) |    # X1 debe ser menor que x1_cuadrilatero
  (df['Y1'] < y1_cuadrilatero) |    # Y1 debe ser menor que y1_cuadrilatero
  (df['X1'] > x4_cuadrilatero) |    # X1 debe ser mayor que x4_cuadrilatero
  (df['Y1'] > y3_cuadrilatero)      # Y1 debe ser mayor que y3_cuadrilatero
)
# Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
condicion2 = (
  (df['X2'] >= x1_cuadrilatero) &   # X2 debe ser mayor o igual que x1_cuadrilatero
  (df['Y2'] >= y1_cuadrilatero) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
  (df['X2'] <= x4_cuadrilatero) &   # X2 debe ser menor o igual que x4_cuadrilatero
  (df['Y2'] <= y3_cuadrilatero)     # Y2 debe ser menor o igual que y3_cuadrilatero
)
# Aplicar las condiciones para filtrar el DataFrame
df = df[condicion1 & condicion2]
dfpasspenareaSP = df[(df['Index'] == 'Assist') | (df['Index'] == 'Key') | (df['Index'] == 'Second assist') | (df['Index'] == 'Complete')].reset_index(drop=True)
dfpasspenareaUP = df[df['Index'] == 'Miss'].reset_index(drop=True)
dfpasspenarea = df.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpasspenarea.columns = ['PlayerID', 'Team', 'Total Passes to Penalty Area']
dfpasspenarea = dfpasspenarea.sort_values('Total Passes to Penalty Area', ascending=False)
dfpasspenareaW = dfpasspenareaSP.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpasspenareaW.columns = ['PlayerID', 'Team', 'Successful Passes to Penalty Area']
dfpasspenareaW = dfpasspenareaW.sort_values(by="Successful Passes to Penalty Area", ascending=False).reset_index(drop=True)
dfpasspenareaF = dfpasspenareaUP.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpasspenareaF.columns = ['PlayerID', 'Team', 'Unsuccessful Passes to Penalty Area']
dfpasspenareaF = dfpasspenareaF.sort_values(by="Unsuccessful Passes to Penalty Area", ascending=False).reset_index(drop=True)
dfpasspenareaA = dfpasspenarea.merge(dfpasspenareaW[['PlayerID', 'Successful Passes to Penalty Area']], on='PlayerID', how='left')
dfpasspenareaB = dfpasspenareaA.merge(dfpasspenareaF[['PlayerID', 'Unsuccessful Passes to Penalty Area']], on='PlayerID', how='left')
dfpasspenareaB = dfpasspenareaB.fillna(0)
dfpasspenareaB['% Successful Passes to Penalty Area'] = round((dfpasspenareaB['Successful Passes to Penalty Area']*100)/dfpasspenareaB['Total Passes to Penalty Area'])
##LONG PASSES
df = df_backup
df['Distancia'] = np.sqrt((df['X2'] - df['X1'])**2 + (df['Y2'] - df['Y1'])**2)
df = df[df['Distancia'] > 30]
dfpasslargesSP = df[(df['Index'] == 'Assist') | (df['Index'] == 'Key') | (df['Index'] == 'Second assist') | (df['Index'] == 'Complete')].reset_index(drop=True)
dfpasslargesUP = df[df['Index'] == 'Miss'].reset_index(drop=True)
dfpasslarges = df.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpasslarges.columns = ['PlayerID', 'Team', 'Total Long Passes']
dfpasslarges = dfpasslarges.sort_values('Total Long Passes', ascending=False)
dfpasslargesW = dfpasslargesSP.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpasslargesW.columns = ['PlayerID', 'Team', 'Successful Long Passes']
dfpasslargesW = dfpasslargesW.sort_values(by="Successful Long Passes", ascending=False).reset_index(drop=True)
dfpasslargesF = dfpasslargesUP.groupby(['PlayerID', 'Team'])['Index'].agg('count').reset_index()
dfpasslargesF.columns = ['PlayerID', 'Team', 'Unsuccessful Long Passes']
dfpasslargesF = dfpasslargesF.sort_values(by="Unsuccessful Long Passes", ascending=False).reset_index(drop=True)
dfpasslargesA = dfpasslarges.merge(dfpasslargesW[['PlayerID', 'Successful Long Passes']], on='PlayerID', how='left')
dfpasslargesB = dfpasslargesA.merge(dfpasslargesF[['PlayerID', 'Unsuccessful Long Passes']], on='PlayerID', how='left')
dfpasslargesB = dfpasslargesB.fillna(0)
dfpasslargesB['% Successful Long Passes'] = round((dfpasslargesB['Successful Long Passes']*100)/dfpasslargesB['Total Long Passes'])
##TOUCHES (FILTER)
df = dfORIGINAL
df_backup2 = df
##TOUCHES
df = df_backup2
dftouches = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftouches.columns = ['PlayerID', 'Team', 'Touches']
dftouches = dftouches.sort_values('Touches', ascending=False)
##TOUCHES IN FINAL THIRD
df = df_backup2
df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)
dftouchesfinthird = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftouchesfinthird.columns = ['PlayerID', 'Team', 'Touches in Final Third']
dftouchesfinthird = dftouchesfinthird.sort_values('Touches in Final Third', ascending=False)
##TOUCHES IN PENALTY AREA
df = df_backup2
df = df[(df['X1'] >= 88.5) & (df['Y1'] >= 13.84) & (df['Y2'] <= 54.16)].reset_index(drop=True)
dftouchespenarea = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftouchespenarea.columns = ['PlayerID', 'Team', 'Touches in Penalty Area']
dftouchespenarea = dftouchespenarea.sort_values('Touches in Penalty Area', ascending=False)
####CARRIES (FILTER)####
df = dfORIGINAL
df = df[(df['Action'] == 'Carries')].reset_index(drop=True)
df_backup3 = df
##TOTAL CARRIES##
df = df_backup3
dftotalcarries = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftotalcarries.columns = ['PlayerID', 'Team', 'Total Carries']
dftotalcarries = dftotalcarries.sort_values('Total Carries', ascending=False)
##CARRIES TO SECOND HALF##
df = df_backup3
df = df[(df['X1'] <= 52.5) & (df['X2'] >= 52.5)].reset_index(drop=True)
dfcarriestohalfpitch = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dfcarriestohalfpitch.columns = ['PlayerID', 'Team', 'Carries to Second Half']
dfcarriestohalfpitch = dfcarriestohalfpitch.sort_values('Carries to Second Half', ascending=False)
##CARRIES TO FINAL THIRD
df = df_backup3
df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)
dfcarriestofinthird = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dfcarriestofinthird.columns = ['PlayerID', 'Team', 'Carries to Final Third']
dfcarriestofinthird = dfcarriestofinthird.sort_values('Carries to Final Third', ascending=False)
##CARRIES TO PENALTY AREA
df = df_backup3
df = df[(df['X1'] >= 88.5) & (df['Y1'] >= 13.84) & (df['Y2'] <= 54.16)].reset_index(drop=True)
dfcarriestopenarea = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dfcarriestopenarea.columns = ['PlayerID', 'Team', 'Carries to Penalty Area']
dfcarriestopenarea = dfcarriestopenarea.sort_values('Carries to Penalty Area', ascending=False)
##SHOTS (FILTER)##
df = dfORIGINAL
df = df[(df['Action'] == 'Shot')].reset_index(drop=True)
df_backup4 = df
##TOTAL SHOTS##
df = df_backup4
dftotalshots = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftotalshots.columns = ['PlayerID', 'Team', 'Total Shots']
dftotalshots = dftotalshots.sort_values('Total Shots', ascending=False)
####DUELS FILTER)####
df = dfORIGINAL
df = df[(df['Action'] == 'Duel')].reset_index(drop=True)
df_backup5 = df
##TOTAL DUELS##
df = df_backup5
dftotalduels = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftotalduels.columns = ['PlayerID', 'Team', 'Total Duels']
dftotalduels = dftotalduels.sort_values('Total Duels', ascending=False)
####AERIAL DUELS FILTER)####
df = dfORIGINAL
df = df[(df['Action'] == 'Aerial duel')].reset_index(drop=True)
df_backup6 = df
##TOTAL AERIAL DUELS##
df = df_backup6
dftotalaerialduels = df.groupby(['PlayerID', 'Team'])['Action'].agg('count').reset_index()
dftotalaerialduels.columns = ['PlayerID', 'Team', 'Total Aerial Duels']
dftotalaerialduels = dftotalaerialduels.sort_values('Total Aerial Duels', ascending=False)
######JOIN DATAFRAMES######
dfTotalA = dfprgB.merge(dfpatofithB[['PlayerID', 'Total Passes to Final Third', 'Successful Passes to Final Third', 'Unsuccessful Passes to Final Third', '% Successful Passes to Final Third']], on='PlayerID', how='outer')
dfTotalB = dfTotalA.merge(dfpasspenareaB[['PlayerID', 'Total Passes to Penalty Area', 'Successful Passes to Penalty Area', 'Unsuccessful Passes to Penalty Area', '% Successful Passes to Penalty Area']], on='PlayerID', how='outer')
dfTotalC = dfTotalB.merge(dfpasslargesB[['PlayerID', 'Total Long Passes', 'Successful Long Passes', 'Unsuccessful Long Passes', '% Successful Long Passes']], on='PlayerID', how='outer')
dfTotalD = dfTotalC.merge(dftouches[['PlayerID', 'Touches']], on='PlayerID', how='outer')
dfTotalE = dfTotalD.merge(dftouchesfinthird[['PlayerID', 'Touches in Final Third']], on='PlayerID', how='outer')
dfTotalF = dfTotalE.merge(dftouchespenarea[['PlayerID', 'Touches in Penalty Area']], on='PlayerID', how='outer')
dfTotalG = dfTotalF.merge(dftotalcarries[['PlayerID', 'Total Carries']], on='PlayerID', how='outer')
dfTotalH = dfTotalG.merge(dfcarriestohalfpitch[['PlayerID', 'Carries to Second Half']], on='PlayerID', how='outer')
dfTotalI = dfTotalH.merge(dfcarriestofinthird[['PlayerID', 'Carries to Final Third']], on='PlayerID', how='outer')
dfTotalJ = dfTotalI.merge(dfcarriestopenarea[['PlayerID', 'Carries to Penalty Area']], on='PlayerID', how='outer')
dfTotalK = dfTotalJ.merge(dftotalshots[['PlayerID', 'Total Shots']], on='PlayerID', how='outer')
dfTotalL = dfTotalK.merge(dftotalduels[['PlayerID', 'Total Duels']], on='PlayerID', how='outer')
dfTotalM = dfTotalL.merge(dftotalaerialduels[['PlayerID', 'Total Aerial Duels']], on='PlayerID', how='outer')

#st.write(dfTotalL)
merged_df = event_counts2.reset_index().merge(dfTotalM, on='PlayerID', how='outer')
df = merged_df
#dfMERGE = df
df = df.fillna(0)

dfplayer = pd.read_excel("MatchesData/all_squads.xlsx")
dfplayer = dfplayer.dropna(subset=['PlayerID'])
#today = datetime.now()
#dfplayer['Age'] = (today - dfplayer['DATE']).astype('<m8[Y]')
#dfplayer['DATE'] = pd.to_datetime(dfplayer['DATE'])
#dfplayer['AgeYear'] = dfplayer['DATE'].str.split('/').str[-1]
#dfplayer 
#st.write(dfplayer['DATE'].astype())
#event_counts = df.groupby(['Players', 'Team'])['Event'].value_counts().unstack(fill_value=0)
columnsevents = df.columns[1:].tolist()

if selected == "Rankings":
    st.title("RANKINGS")
    st.markdown("""----""")
    df = df.merge(dfplayer[['PlayerID', 'NAME', 'TEAM', 'POSITION', 'MINUTES PLAYED', 'YEAR']], on='PlayerID', how='outer')
    column_order = ['PlayerID', 'NAME', 'TEAM', 'POSITION', 'MINUTES PLAYED', 'YEAR'] + [col for col in df.columns if col not in ['PlayerID', 'NAME', 'TEAM', 'POSITION', 'MINUTES PLAYED', 'YEAR']]
    # Reorganiza las columnas del DataFrame
    df = df[column_order]
    df = df.rename(columns={'POSITION': 'Position', 'MINUTES PLAYED': 'Minutes Played', 'YEAR': 'Year', 'NAME': 'Name'})
    st.write(df)
 
    but0, but1 = st.columns(2)
    with but0:
        name = "MetricsData_FutureTalentsVII"
        df_xlsx = to_excel(df)
        st.download_button(label='Descargar Archivo Excel',
                           data=df_xlsx,
                           file_name= ""+ name +".xlsx")

    with but1:
        df_csv = convert_df(df)
        st.download_button(label="Descargar Archivo CSV",
                           data=df_csv,
                           file_name=""+ name +".csv",
                           mime='text/csv')
    st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
    #st.write(len(event_counts))
    st.markdown("""----""")
    metricsearchbox01, metricsearchbox02, metricsearchbox03 = st.columns(3)
    GroupOpt_Defensive = ['Aerial duel - Lost', 'Aerial duel - Won', 'Allow crosses - ', 'Anticipation - Complete', 'Anticipation - Half', 'Block - Cross', 'Block - Shot', 'Clearance - ', 'Coverage - Complete', 'Coverage - Half', 'Interception - Complete', 'Interception - Half', 'Tackles - Lost', 'Tackles - Won', 'Recovery - ', 'Loses the mark - Normal', 'Loses the mark - Severity']
    GroupOpt_Offensive = ['Total Shots', 'Shot - Goal', 'Shot - On target', 'Shot - Wide', 'Shot - Post', 'Touches in Penalty Area']
    GroupOpt_Possesion = ['Total Duels', 'Duel - Won', 'Duel - Lost', 'Total Aerial Duels', 'Touches', 'Touches in Final Third', 'Total Carries', 'Carries to Second Half', 'Carries to Final Third', 'Carries to Penalty Area', 'Carries - Ball', 'Carries - To space', 'Take-ons - Won', 'Take-ons - Lost', 'Received pass']
    GroupOpt_Distribut = ['Pass - Complete', 'Pass - Miss', 'Type pass - Assist', 'Type pass - Key', 'Type pass - Second assist', 'Total Progressive Passes', 'Successful Progressive Passes', 'Unsuccessful Progressive Passes', '% Successful Progressive Passes', 'Total Passes to Final Third', 'Successful Passes to Final Third', 'Unsuccessful Passes to Final Third', '% Successful Passes to Final Third', 'Total Passes to Penalty Area', 'Successful Passes to Penalty Area', 'Unsuccessful Passes to Penalty Area', '% Successful Passes to Penalty Area', 'Total Long Passes', 'Successful Long Passes', 'Unsuccessful Long Passes', '% Successful Long Passes']
    GroupOpt_SetPieces = ['Corner - Complete', 'Corner - Miss', 'Free kick - Complete', 'Free kick - Miss', 'Free kick - Shot', 'Throw-in - ', 'Throw-in - Complete', 'Throw-in - Miss']
    with metricsearchbox01:
      MetricGroupOption = ['Offensive', 'Defensive', 'Possesion', 'Distribution', 'Creation', 'Set Pieces']
      MetricGroupSel = st.selectbox("Choose metrics group:", MetricGroupOption)
      if MetricGroupSel == "Defensive":
       Metric_Lst = GroupOpt_Defensive
      elif MetricGroupSel == "Offensive":
       Metric_Lst = GroupOpt_Offensive
      elif MetricGroupSel == "Possesion":
       Metric_Lst = GroupOpt_Possesion
      elif MetricGroupSel == "Distribution":
       Metric_Lst = GroupOpt_Distribut
      elif MetricGroupSel == "Set Pieces":
       Metric_Lst = GroupOpt_SetPieces
      else:
       Metric_Lst = columnsevents 
    with metricsearchbox02:
        #Team_Lst = df['Team'].drop_duplicates().tolist()
        MetricSel = st.selectbox("Choose metric:", Metric_Lst)
        #event_counts = event_counts[event_counts[MetricSel]].reset_index(drop=True)
    with metricsearchbox03:
         #minsel = st.slider('Filter by minutes (%):', 0, 100)
         #maxmin = df['Minutes Played'].max() + 5
         #minsel1 = (minsel*maxmin)/100
         #df = df[df['Minutes Played'] >= minsel1].reset_index(drop=True)
        metricsearbox03_01, metricsearbox03_02 = st.columns(2)
        with metricsearbox03_01:
         minsel = st.slider('Filter by minutes (%):', 0, 100)
         maxmin = df['Minutes Played'].max() + 5
         minsel1 = (minsel*maxmin)/100
         df = df[df['Minutes Played'] >= minsel1].reset_index(drop=True)
        with metricsearbox03_02:
         #df['Year'] = df['Year'].astype(str)
         #agemin = str(df['Year'].min())
         #agemax = str(df['Year'].max())
         agesel = st.slider('Filter by Year:', round(df['Year'].min()), round(df['Year'].max()), (round(df['Year'].min()), round(df['Year'].max())), 1)
         #df = df[df['Year'] >= agesel].reset_index(drop=True)
         df = df[df['Year'] <= agesel[1]]
         df = df[df['Year'] >= agesel[0]]
     
      #df = df[df['Players'] == PlayerSel].reset_index(drop=True)
    p01, p02 = st.columns(2)
    with p01:
        ##fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
        ##ax.axis("off")
        ##fig.patch.set_visible(False)
        fig, ax = mplt.subplots(figsize = (10,13), dpi=600)
        fig.set_facecolor('#151517')
        ax.patch.set_facecolor('#151517')
        df = df.sort_values(by=[MetricSel], ascending=True)
        #players_teams = [f'{player} - {team}' for player, team in event_counts[-10:].index]
        #events = event_counts[MetricSel].head(-5)
        #players_teams = players_teams.str.upper()
        ##st.write(event_counts.columns)
        ##st.write(players_teams)
        #event_counts = event_counts.head(10)
        colors = colorlist((1, 0, 0.3137254901960784, 0.5), (1, 0, 0.3137254901960784, 1), 10)
        df000 = df
        PLY = df['PlayerID'].tail(10).str.upper()
        Z = df[MetricSel].tail(10)
        ax.barh(PLY, Z, edgecolor=(1,1,1,0.5), lw = 1, color=colors)
        mplt.setp(ax.get_yticklabels(), fontproperties=prop2, fontsize=18, color='#FFF')
        mplt.setp(ax.get_xticklabels(), fontproperties=prop2, fontsize=20, color=(1,1,1,1))
        mplt.xlabel(MetricSel.upper(), color = 'w', fontproperties=prop2, fontsize=22, labelpad=30)
        #ax.set_xticks([0, 5, 10])
        #ax.set_xlim(0, 18)
        ax.tick_params(axis='y', which='major', pad=15)
        spines = ['top','bottom','left','right']
        for x in spines:
            if x in spines:
                ax.spines[x].set_visible(False)
        st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    with p02:
        event_counts22 = df000.sort_values(by=[MetricSel], ascending=False)
        #st.write(event_counts22[['PlayerID', MetricSel]])
        st.write(event_counts22[['PlayerID', 'Minutes Played', 'Year', MetricSel]])
st.divider()
if selected == "Player Search":
    ###Data
    ##df = pd.read_excel("MatchesData/matches.xlsx")
    df = pd.read_excel("MatchesData/all_matches_prueba4.xlsx")
    st.title("Search Player")
    
    #df['FieldXfrom'] = (df['FieldXfrom']*105)/1
    #df['FieldYfrom'] = (df['FieldYfrom']*68)/1
    #df['FieldXto'] = (df['FieldXto']*105)/1
    #df['FieldYto'] = (df['FieldYto']*68)/1

    df['X1'] = (df['X1']*105)/1
    df['Y1'] = (df['Y1']*68)/1
    df['X2'] = (df['X2']*105)/1
    df['Y2'] = (df['Y2']*68)/1
    ###Convert for left attack-side
    for index, row in df.iterrows():
        if row['Atack side'] == 'Left':
            df.at[index, 'X1'] = 105 - row['X1']
            df.at[index, 'Y1'] = 68 - row['Y1']
            df.at[index, 'X2'] = 105 - row['X2']
            df.at[index, 'Y2'] = 68 - row['Y2']
    df_shots = df[df['Action'] == 'Shot'].reset_index(drop=True)
    teamsearchbox01, teamsearchbox02, teamsearchbox03 = st.columns(3)
    with teamsearchbox01:
      Team_Lst = df['Team'].drop_duplicates().tolist()
      TeamSel = st.selectbox("Choose team:", Team_Lst)
      df = df[df['Team'] == TeamSel].reset_index(drop=True)
    with teamsearchbox02:
      Positions_List = df['Position'].drop_duplicates().tolist()
      PositionsSel = st.selectbox("Choose position:", Positions_List)
      df = df[df['Position'] == PositionsSel].reset_index(drop=True)
    with teamsearchbox03:
      Player_Lst = df['Players'].drop_duplicates().tolist()
      PlayerSel = st.selectbox("Choose player:", Player_Lst)
      df = df[df['Players'] == PlayerSel].reset_index(drop=True)
    st.markdown("""----""")  
    st.markdown("<style> div { text-align: left } </style>", unsafe_allow_html=True)
    st.header(PlayerSel)
    st.subheader(TeamSel)
    
    css='''
    [data-testid="metric-container"] {
        width: fit-content;
        margin: auto;
    }
    
    [data-testid="metric-container"] > div {
        width: fit-content;
        margin: auto;
    }
    
    [data-testid="metric-container"] label {
        width: fit-content;
        margin: auto;
    }
    '''
    st.markdown(f'<style>{css}</style>',unsafe_allow_html=True)
    st.markdown("""----""")
    st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
    st.title("ACTIONS")
    with st.form(key='formpltev'):
        pltev01, pltev02, pltev03 = st.columns(3)
        with pltev01:
            Eventlst = ['Actions', 'Passes', 'Shots', 'Dribbles', 'Recoveries', 'Pressures']
            EventlstSel = st.selectbox('Choose metric:', Eventlst)
            #st.dataframe(dfDOWN)
    #with pltev02:
        #Typelst = ['Mapa de Acciones', 'Territorio de Acciones', 'Mapa de Calor Acciones - JdP', 'Mapa de Calor Acciones - Bins']
        #st.selectbox('Seleccionar tipo gráfico:', Typelst)
        with pltev02:
            LstTeam = df['Team'].drop_duplicates()
            LstTeamsel = st.selectbox('Choose team:', LstTeam)
            #df = df[df['Team'] == LstTeamsel].reset_index(drop=True)
            #st.dataframe(dfDOWN)
        with pltev03:
            LstPlayer = df['Players'].drop_duplicates()
            LstPlayer = LstPlayer.tolist()
            PlayerPltSel = st.selectbox('Choose player:', LstPlayer)
            #df = df[df['Players'] == PlayerPltSel].reset_index(drop=True)
            #st.dataframe(dfDOWN)
        submit_button_pltev = st.form_submit_button(label='OK')
    
    #selbox01, selbox02, selbox03 = st.columns(3)
    #with selbox01:
    #  Lista_Partidos = ['Fecha 1', 'Fecha 2']
    #  st.selectbox("Choose matchday:", Lista_Partidos) 
    #with selbox02:
    #  Team_Lst = df['Team'].drop_duplicates().tolist()
    #  TeamSel = st.selectbox("Choose team:", Team_Lst)
    #  #df = df[df['Team'] == TeamSel].reset_index(drop=True)
    #with selbox03:
    #  Player_Lst = df['Players'].drop_duplicates().tolist()
    #  PlayerSel = st.selectbox("Choose player:", Player_Lst)
    #  #df = df[df['Players'] == PlayerSel].reset_index(drop=True)
    ##df = df.rename(columns={'FieldXfrom': 'X1',
    ##                                'FieldYfrom': 'Y1',
    ##                                'FieldXto': 'X2',
    ##                                'FieldYto': 'Y2'})
    dfORIGINAL = df
    if EventlstSel == 'Actions':     
        pltmnop01, pltmnop02, pltmnop03 = st.columns(3)
        with pltmnop01:
            OptionPlot = ['Territory Actions', 'Heatmap - Zones', 'Heatmap - Gaussian', 'Heatmap - Kernel']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        #with pltmnop02:
            #EfectMinSel = st.slider('Seleccionar rango de partido:', 0, MaxAddMin, (0, MaxAddMin))
            #EfectMinSel = st.slider('Seleccionar rango de partido:', 0, 90, (0, 90))
        #with pltmnop03:
            #ColorOptionSel = st.color_picker('Choose color:', '#FF0046')
        ColorOptionSel = "#FF0050"
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ##ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Attack Direction', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            ###df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
            
            ##st.write(df)
            ##ax.scatter(df['FieldXfrom'], df['FieldYfrom'], color = "#FF0046", edgecolors='w', s=30, zorder=2, alpha=0.2)
            ##st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            if OptionPlotSel == 'Territory Actions': 
                    
                    ##df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                    ##dfKKcleaned = df
                    
                    ##df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                    dfKKcleaned = df
                    scaler  = StandardScaler()
                    
                    
                    defpoints1 = df[['X1', 'Y1']].values
                    defpoints2 = scaler.fit_transform(defpoints1)
                    df2 = pd.DataFrame(defpoints2, columns = ['Xstd', 'Ystd'])
                    df3 = pd.concat([df, df2], axis=1)
                    df5=df3
                    df3 = df3[df3['Xstd'] <= 1]
                    df3 = df3[df3['Xstd'] >= -1]
                    df3 = df3[df3['Ystd'] <= 1]
                    df3 = df3[df3['Ystd'] >= -1].reset_index()
                    df9 = df
                    df = df3
                    defpoints = df[['X1', 'Y1']].values
                    #st.write(defpoints)
    
                    hull = ConvexHull(df[['X1','Y1']])        
                    ax.scatter(df9['X1'], df9['Y1'], color = ColorOptionSel, edgecolors='w', s=30, zorder=2, alpha=0.2)
                    #Loop through each of the hull's simplices
                    for simplex in hull.simplices:
                        #Draw a black line between each
                        ax.plot(defpoints[simplex, 0], defpoints[simplex, 1], '#BABABA', lw=2, zorder = 1, ls='--')
                    ax.fill(defpoints[hull.vertices,0], defpoints[hull.vertices,1], ColorOptionSel, alpha=0.7)
                    meanposx = df9['X1'].mean()
                    meanposy = df9['Y1'].mean()
                    ax.scatter(meanposx, meanposy, s=1000, color="w", edgecolors=ColorOptionSel, lw=2.5, zorder=25, alpha=0.95)
                    names = PlayerPltSel.split()
                    iniciales = ""
                    for name in names:
                       iniciales += name[0] 
                    #names_iniciales = names_iniciales.squeeze().tolist()
                    ax.text(meanposx, meanposy, iniciales, color='k', fontproperties=prop2, fontsize=13, zorder=34, ha='center', va='center')
                    ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(dfKKcleaned)) + " TOUCHES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                    #Adding title
                    ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
                    ax9.scatter(2, 5, s=120, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1)
                    ax9.text(2, -0.5, 'TOUCHES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                    ax9.scatter(8, 5, s=320, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                    ax9.text(8, -0.5, 'ACTIONS\nTERRITORY', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            elif OptionPlotSel == 'Heatmap - Zones':
    
                    #df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                    dfKKcleaned = df
                    # Definir los colores base con transparencias diferentes
                    red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
                    green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
                    blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
                    # Crear una lista de los colores y las posiciones en el colormap
                    colors = [red, green, blue]
                    positions = [0, 0.5, 1]
                    # Crear el colormap continuo con transparencias
                    cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
                    path_eff = [path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]
                    bin_statistic = pitch.bin_statistic_positional(df.X1, df.Y1, statistic='count', positional='full', normalize=True)
                    pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmaps, edgecolors='#524F50', linewidth=1)
                    pitch.scatter(df.X1, df.Y1, c='w', s=15, alpha=0.02, ax=ax)
                    labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14, fontproperties=prop2, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
                    ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(dfKKcleaned)) + " TOUCHES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                    ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                    ax9.scatter(6.75,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                    ax9.scatter(5.00,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                    ax9.scatter(3.25,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                    ax9.text(5, 0, '-  TOUCHES  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
    
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            elif OptionPlotSel == 'Heatmap - Gaussian':
                    #df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                    dfKKcleaned = df
                    # Definir los colores base con transparencias diferentes
                    red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.3]   # Rojo opaco
                    green = [1, 0, 0.2745098039215686, 1]   # Verde semitransparente
                    blue = [1, 0.5490196078431373, 0.6745098039215686, 1]   # Azul semitransparente    
                    # Crear una lista de los colores y las posiciones en el colormap
                    colors = [red, green, blue]
                    positions = [0, 0.5, 1]
                    # Crear el colormap continuo con transparencias
                    cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
                    bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                    ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(dfKKcleaned)) + " TOUCHES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                    ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                    ax9.scatter(6.75,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                    ax9.scatter(5.00,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                    ax9.scatter(3.25,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                    ax9.text(5, 0, '-  TOUCHES  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            elif OptionPlotSel == 'Heatmap - Kernel':
                    #df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                    dfKKcleaned = df
                    # Definir los colores base con transparencias diferentes
                    red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.3]   # Rojo opaco
                    green = [1, 0, 0.2745098039215686, 1]   # Verde semitransparente
                    blue = [1, 0.5490196078431373, 0.6745098039215686, 1]   # Azul semitransparente    
                    # Crear una lista de los colores y las posiciones en el colormap
                    colors = [red, green, blue]
                    positions = [0, 0.5, 1]
                    # Crear el colormap continuo con transparencias
                    cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
                    #bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                    #bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                    kde = pitch.kdeplot(dfKKcleaned.X1, dfKKcleaned.Y1, ax=ax,
                        # fill using 100 levels so it looks smooth
                        fill=True, levels=500,
                        # shade the lowest area so it looks smooth
                        # so even if there are no events it gets some color
                        thresh=0,
                        cut=1, alpha=0.7, zorder=-2,  # extended the cut so it reaches the bottom edge
                        cmap=cmaps)
    
                    
                    #pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                    ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(dfKKcleaned)) + " TOUCHES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                    ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                    ax9.scatter(6.75,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                    ax9.scatter(5.00,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                    ax9.scatter(3.25,5, c=ColorOptionSel, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                    ax9.text(5, 0, '-  TOUCHES  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        with pltmain02:
            st.dataframe(df)
    elif EventlstSel == 'Passes':
        pltmnop11, pltmnop12, pltmnop13 = st.columns(3)
        with pltmnop11:
            OptionPlot = ['Passes Map', 'Progressive Passes Map', 'Passes to Final Third Map', 'Passes to Penalty Area', 'xT Passes Map']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        #with pltmnop12:
            #EfectMinSel = st.slider('Seleccionar rango de partido:', 0, 90, (0, 90))
        with pltmnop12:
                MetOption = ['Pases Claves', 'Asistencias']
                MetOptionSel = st.selectbox('Choose metric:', MetOption)
        pltmain11, pltmain12 = st.columns(2)
        with pltmain11:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Attack Direction', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            ##st.dataframe(dfDOWN)
            ##df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            df = dfORIGINAL
            df_backup = df
            colorviz="#FF0046"
            if OptionPlotSel == 'Passes Map':
                ##df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index()
                dfKKK = df
                df = df_backup
                df = df[(df['Action'] == 'Pass') | (df['Action'] == 'Type pass')].reset_index(drop=True)
                ##dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                ##dfast = df[df['Event'] == 'Assists']
                ##dfkey = df[df['Event'] == 'Key Passes']
                ##dfpas = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')]
                dfpas = df[df['Action'] == 'Pass'].reset_index(drop=True)
                dfkey = df[df['Action'] == 'Type pass'].reset_index(drop=True)
                
                ###Progressive
                df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
                df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
                df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.8 for x in range(len(df.Beginning))]
                
                
                ###Filter by passes progressives
                dfprog = df[df['Progress'] == True].reset_index()
                dfprog = dfprog.drop(['index'], axis=1)    
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                x1 = dfpas['X1']
                y1 = dfpas['Y1']
                x2 = dfpas['X2']
                y2 = dfpas['Y2']
                
                x1a = dfprog['X1']
                y1a = dfprog['Y1']
                x2a = dfprog['X2']
                y2a = dfprog['Y2']
                
                x1k = dfkey['X1']
                y1k = dfkey['Y1']
                x2k = dfkey['X2']
                y2k = dfkey['Y2']

                pitch.lines(x1, y1, x2, y2, cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True) 
                ax.scatter(x2, y2, color='#9F9F9F', edgecolors='#121214', zorder=3, lw=0.5)       
                    
                pitch.lines(x1a, y1a, x2a, y2a, cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(x2a, y2a, color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)           
                
                pitch.lines(x1k, y1k, x2k, y2k, cmap=get_continuous_cmap(hex_list1), ax=ax, lw=2, comet=True, transparent=True, zorder=10) 
                ax.scatter(x2k, y2k, color="#C7B200", edgecolors='#121214', zorder=5, lw=0.5)
                ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " SUCCESSFUL PASSES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                ax9.text(26.25, 2, 'SUCCESSFUL\nPASSES', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(52.5, 12, marker='s', color=colorviz, s=300)
                ax9.text(52.5, 2, 'PROGRESSIVE\nPASSES', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(78.75, 12, marker='s', color='#C7B200', s=300)
                ax9.text(78.75, 2, 'KEY\nPASSES', color='#C7B200', fontproperties=prop2, ha='center', fontsize=9)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 

            if OptionPlotSel == 'Progressive Passes Map':
                ##df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                ##dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                ###Progressive
                dfKKK = df
                df = df_backup
                df = df[(df['Action'] == 'Pass') | (df['Action'] == 'Type pass')].reset_index(drop=True)
                dfpas = df[df['Action'] == 'Pass'].reset_index(drop=True)
                dfkey = df[df['Action'] == 'Type pass'].reset_index(drop=True)
                df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
                df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
                df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.8 for x in range(len(df.Beginning))]
                                          
                ###Filter by passes progressives
                dfprog = df[df['Progress'] == True].reset_index()
                dfprog = dfprog.drop(['index'], axis=1)
                ##dfprog = dfprog.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                dfw = dfprog[(dfprog['Index'] == 'Complete') | (dfprog['Index'] == 'Assists') | (dfprog['Index'] == 'Key') | (dfprog['Index'] == 'Second assist')].reset_index(drop=True)
                dff = dfprog[(dfprog['Index'] == 'Miss')].reset_index(drop=True)
                ##dfw = dfprog[(dfprog['Event'] == 'Successful passes') | (dfprog['Event'] == 'Key Passes') | (dfprog['Event'] == 'Assists') | (dfprog['Event'] == 'Successful open play crosses') | (dfprog['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                ##dff = dfprog[(dfprog['Event'] == 'Unsuccessful passes') | (dfprog['Event'] == 'Unsuccessful open play crosses') | (dfprog['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)

                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  

                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)     
                ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(dfprog)) + " PROGRESSIVE PASSES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'SUCCESSFUL\nPASSES', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'UNSUCCESSFUL\nPASSES', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                df = dfprog
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
            if OptionPlotSel == 'Passes to Final Third Map':
                ##df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                ##df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)
                df = df_backup
                df = df[(df['Action'] == 'Pass') | (df['Action'] == 'Type pass')].reset_index(drop=True)
                df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)
                ##dfwin = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                ##dffail = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                dfwin = df[(df['Index'] == 'Complete') | (df['Index'] == 'Assists') | (df['Index'] == 'Key') | (df['Index'] == 'Second assist')].reset_index(drop=True)
                dffail = df[(df['Index'] == 'Miss')].reset_index(drop=True)
                dfKKK = df
                ##dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                pitch.lines(dfwin['X1'], dfwin['Y1'], dfwin['X2'], dfwin['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True)
                ax.scatter(dfwin['X2'], dfwin['Y2'], color='#FF0050', edgecolors='#121214', zorder=3, lw=0.5)
                pitch.lines(dffail['X1'], dffail['Y1'], dffail['X2'], dffail['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True)
                ax.scatter(dffail['X2'], dffail['Y2'], color='#9F9F9F', edgecolors='#121214', zorder=3, lw=0.5)
                ax.vlines(x=70, ymin=0, ymax=68, color='w', alpha=0.3, ls='--', zorder=-1)
                ax.add_patch(Rectangle((70, 0), 35, 68, fc="#000000", fill=True, alpha=0.7, zorder=-2))

                ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " PASSES TO FINAL THIRD", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'SUCCESSFUL\nPASSES', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'UNSUCCESSFUL\nPASSES', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
                #pitch.lines(x1a, y1a, x2a, y2a, cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                #ax.scatter(x2a, y2a, color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)
            if OptionPlotSel == 'Passes to Penalty Area':
                #df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                #dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                df = df_backup
                df = df[(df['Action'] == 'Pass') | (df['Action'] == 'Type pass')].reset_index(drop=True)
                # Coordenadas del cuadrilátero
                x1_cuadrilatero, y1_cuadrilatero = 88.5, 13.84
                x2_cuadrilatero, y2_cuadrilatero = 105, 13.84
                x3_cuadrilatero, y3_cuadrilatero = 88.5, 54.16
                x4_cuadrilatero, y4_cuadrilatero = 105, 54.16
                
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1 = (
                    (df['X1'] < x1_cuadrilatero) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero)      # Y1 debe ser mayor que y3_cuadrilatero
                )
                
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2 = (
                    (df['X2'] >= x1_cuadrilatero) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
                
                # Aplicar las condiciones para filtrar el DataFrame
                df = df[condicion1 & condicion2]
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)

                #dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                #dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                dfw = df[(df['Index'] == 'Complete') | (df['Index'] == 'Assists') | (df['Index'] == 'Key') | (df['Index'] == 'Second assist')].reset_index(drop=True)
                dff = df[(df['Index'] == 'Miss')].reset_index(drop=True)
                
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  

                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)  
                ax.vlines(x=88.5, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.vlines(x=105, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=88.5, xmax=105, y=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=88.5, xmax=105, y=13.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " PASSES TO PENALTY AREA", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'SUCCESSFUL\nPASSES', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'UNSUCCESSFUL\nPASSES', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
            if OptionPlotSel == 'xT Passes Map':
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
        with pltmain12:
            st.dataframe(df)
    if EventlstSel == 'Recoveries':     
        pltmnop01_recoveries, pltmnop02_recoveries, pltmnop03_recoveries = st.columns(3)
        with pltmnop01_recoveries:
            OptionPlot = ['Recoveries Location', 'Heatmap - Zones']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        ColorOptionSel = "#FF0050"
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ##ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Attack Direction', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            ###df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            df = dfORIGINAL
            dfKK = df
            df_backup = df
            if OptionPlotSel == 'Recoveries Location': 

                    dfKKcleaned = df
                    df = df[df['Action'] == 'Recovery'].reset_index(drop=True)
                    #scaler  = StandardScaler()
                    #defpoints1 = df[['X1', 'Y1']].values
                    #defpoints2 = scaler.fit_transform(defpoints1)
                    #df2 = pd.DataFrame(defpoints2, columns = ['Xstd', 'Ystd'])
                    #df3 = pd.concat([df, df2], axis=1)
                    #df5=df3
                    #df3 = df3[df3['Xstd'] <= 1]
                    #df3 = df3[df3['Xstd'] >= -1]
                    #df3 = df3[df3['Ystd'] <= 1]
                    #df3 = df3[df3['Ystd'] >= -1].reset_index()
                    #df9 = df
                    #df = df3
                    #defpoints = df[['X1', 'Y1']].values
                    #hull = ConvexHull(df[['X1','Y1']])        
                    #ax.scatter(df9['X1'], df9['Y1'], color = ColorOptionSel, edgecolors='w', s=30, zorder=2, alpha=0.2)
                    ##Loop through each of the hull's simplices
                    #for simplex in hull.simplices:
                    #    #Draw a black line between each
                    #    ax.plot(defpoints[simplex, 0], defpoints[simplex, 1], '#BABABA', lw=2, zorder = 1, ls='--')
                    #ax.fill(defpoints[hull.vertices,0], defpoints[hull.vertices,1], ColorOptionSel, alpha=0.7)
                    #meanposx = df9['X1'].mean()
                    #meanposy = df9['Y1'].mean()
                    #ax.scatter(meanposx, meanposy, s=1000, color="w", edgecolors=ColorOptionSel, lw=2.5, zorder=25, alpha=0.95)
                    #names = PlayerPltSel.split()
                    #iniciales = ""
                    #for name in names:
                    #   iniciales += name[0] 
                    ##names_iniciales = names_iniciales.squeeze().tolist()
                    #ax.text(meanposx, meanposy, iniciales, color='k', fontproperties=prop2, fontsize=13, zorder=34, ha='center', va='center')
                    ax.scatter(df['X1'], df['Y1'], color = ColorOptionSel, edgecolors='w', s=30, zorder=2, alpha=0.2)
                    ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " RECOVERIES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                    ##Adding title
                    ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
                    ax9.scatter(2, 5, s=120, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1)
                    ax9.text(2, -0.5, 'RECOVERIES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                    #ax9.scatter(8, 5, s=320, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                    #ax9.text(8, -0.5, 'RECOVERIES\nTERRITORY', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        with pltmain02:
            st.dataframe(df)
    elif EventlstSel == 'Shots':
        pltmnop01_shots, pltmnop02_shots, pltmnop03_shots = st.columns(3)
        with pltmnop01_shots:
            OptionPlot = ['Shots Location', 'Shots Heatmap']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        with pltmnop02_shots:
            #df_pltmnop02shots = dfORIGINAL[dfORIGINAL['Action'] == 'Shot'].reset_index(drop=True)
            IndexOption = df_shots['Index'].drop_duplicates().tolist()
            IndexSel = st.multiselect('Choose type:', IndexOption)
            
        with pltmnop03_shots:
            Sel = st.slider('Expected Goals (xG):', 0, 10, (0, 10))
        ColorOptionSel = "#FF0050"
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            #st.dataframe(dfDOWN)
            ###df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            df = dfORIGINAL
            dfKK = df
            df_backup = df
            if OptionPlotSel == "Shots Location": 
                    #df = df[df['Index'] == IndexSel].reset_index(drop=True)
                    df = df[df['Index'].isin(IndexSel)].reset_index(drop=True)
                    fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
                    ax.axis("off")
                    fig.patch.set_visible(False)
                    pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
                    pitch.draw(ax=ax)
                    #Adding directon arrow
                    ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
                    ax29.axis("off")
                    ax29.set_xlim(0,10)
                    ax29.set_ylim(0,10)
                    ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                    ##ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                    ax29.text(5, 2, 'Attack Direction', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
                    #Adding winstats logo
                    ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
                    url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
                    response = requests.get(url53)
                    img = Image.open(BytesIO(response.content))
                    ax53.imshow(img)
                    ax53.axis("off")
                    ax53.set_facecolor("#000")
                    dfKKcleaned = df
                    df = df[df['Action'] == 'Shot'].reset_index(drop=True)
                    ax.scatter(df['X1'], df['Y1'], color = ColorOptionSel, edgecolors='w', s=30, zorder=2, alpha=1)
                    ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " SHOTS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                    ##Adding title
                    ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
                    ax9.scatter(2, 5, s=120, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1)
                    ax9.text(2, -0.5, 'SHOTS\nLOCATION', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                    #ax9.scatter(8, 5, s=320, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                    #ax9.text(8, -0.5, 'RECOVERIES\nTERRITORY', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            if OptionPlotSel == "Shots Heatmap":
                    fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
                    ax.axis("off")
                    fig.patch.set_visible(False)
                    soc_pitch_divisions(ax, grids = True)
                    df = df_backup
                    df = df[df['Action'] == 'Shot'].reset_index(drop=True)
                    df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                    y_bins = [105] + [105 - 5.5*x for x in range(1,10)] + [45]
                    x_bins = [68] + [68 - 6.8*x for x in range(1,10)] + [0]
                    x_bins.sort()
                    y_bins.sort()
                    
                    df["bins_x"] = pd.cut(df["X1"], bins = x_bins)
                    df["bins_y"] = pd.cut(df["Y1"], bins = y_bins)
                    ##Group and sum xGOT by side and location
                    df_teams = (
                        df.groupby(
                            ["bins_x", "bins_y"], 
                            observed = True
                        )["Action"].count()
                        .reset_index()
                    )
                    
                    # And we sort it based on the bins_y and bins_x columns
                    df_teams = (
                        df_teams.
                        sort_values(by = ["bins_y", "bins_x"]).
                        reset_index(drop = True)
                    )
                                        
                    example_df = df_teams
                    total_example = example_df["Action"].sum()
                    
                    # Compute share of xGOT as a % of total
                    example_df = (
                        example_df
                        .assign(COUNT_share = lambda x: x.Action/total_example)
                    )
                    # Scale data to the maximum value to get a nice color scale
                    example_df = (
                        example_df
                        .assign(COUNT_scaled = lambda x: x.COUNT_share/x.COUNT_share.max())
                    )
                    
                    counter = 0
                    for X, Y in zip(example_df["bins_x"], example_df["bins_y"]):
                    	#This colours our bins
                        ax.fill_between(
                            x = [X.left, X.right],
                            y1 = Y.left,
                            y2 = Y.right,
                            color = "#FF0046",
                            alpha = example_df["COUNT_scaled"].iloc[counter],
                            zorder = -1,
                            lw = 0
                        )
                    	
                        # Fancy annotations cuz why not?
                        if example_df['COUNT_share'].iloc[counter] > .005:
                            text_ = ax.annotate(
                                xy = (X.right - (X.right - X.left)/2, Y.right - (Y.right - Y.left)/2),
                                text = f"{example_df['COUNT_share'].iloc[counter]:.0%}",
                                fontproperties=prop2,
                                ha = "center",
                                va = "center",
                                color = "w",
                                size = 14,
                                weight = "bold",
                                zorder = 3
                            )
                    
                            text_.set_path_effects(
                                [path_effects.Stroke(linewidth=0.75, foreground="k"), path_effects.Normal()]
                            )
                    
                        counter += 1
                    auxtitle = ax.text(34, 110, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " SHOTS",c='w', fontproperties=prop2, fontsize=11, ha='center', va='center')
                    ##Adding winstats logo
                    ax53 = fig.add_axes([0.82, 0.12, 0.05, 0.05])
                    url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
                    response = requests.get(url53)
                    img = Image.open(BytesIO(response.content))
                    ax53.imshow(img)
                    ax53.axis("off")
                    ax53.set_facecolor("#000")
                    #Adding colorbar
                    ax9 = fig.add_axes([0.12,0.12,0.20,0.07])
                    ax9.scatter(6.75,5, c="#FF0050", marker='h', s=400, edgecolors='#121214', alpha=1)
                    ax9.scatter(5.00,5, c="#FF0050", marker='h', s=400, edgecolors='#121214', alpha=0.7)
                    ax9.scatter(3.25,5, c="#FF0050", marker='h', s=400, edgecolors='#121214', alpha=0.2)
                    ax9.text(5, 0.5, '-  SHOTS  +',c='w', fontproperties=prop2, fontsize=9.5, ha='center')
                    ax9.axis("off")
                    ax9.set_xlim(0,10)
                    ax9.set_ylim(0,10)
                    st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        with pltmain02:
            st.dataframe(df)
    elif EventlstSel == 'Pressures':
        pltmnop01_pressures, pltmnop02_pressures, pltmnop03_pressures = st.columns(3)
        with pltmnop01_pressures:
            OptionPlot = ['Pressures Location', 'Heatmap - Zones']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        ColorOptionSel = "#FF0050"
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ##ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Attack Direction', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            ###df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            df = dfORIGINAL
            dfKK = df
            df_backup = df
            if OptionPlotSel == 'Pressures Location':
             dfKKcleaned = df
             df = df[df['Action'] == 'Pressure'].reset_index(drop=True)
             ax.scatter(df['X1'], df['Y1'], color = ColorOptionSel, edgecolors='w', s=30, zorder=2, alpha=0.2)
             ax.text(52.5,70, "" + PlayerPltSel.upper() + " - " + str(len(df)) + " PRESSURES", c='w', fontsize=10, fontproperties=prop2, ha='center')
             ##Adding title
             ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
             ax9.axis("off")
             ax9.set_xlim(0,10)
             ax9.set_ylim(0,10)
             ax9.scatter(2, 5, s=120, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1)
             ax9.text(2, -0.5, 'PRESSURES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
             #ax9.scatter(8, 5, s=320, color=ColorOptionSel, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
             #ax9.text(8, -0.5, 'RECOVERIES\nTERRITORY', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
             st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        with pltmain02:
             st.dataframe(df)
             
    st.divider()
    st.dataframe(df)
    ###SEGMENTAR POR GRUPO DE MÉTRICAS###
    df = pd.read_excel("MatchesData/MetricsData_FutureTalentsVII.xlsx")
    #df = df[df['P
    dfofe = df[['Total Shots', 'Shot - Goal', 'Shot - On target', 'Touches in Penalty Area']]
    dfofel = dfofe.columns
    dfpos = df[['Total Duels', 'Total Aerial Duels', 'Touches', 'Total Carries', 'Received pass - ']]
    dfposl = dfpos.columns
    valuessofe = dfofe.iloc[0,:]
    valuessofe2 = round(dfofe.mean(), 2)
    lowwofe = []
    highhofe = []
    for an in range(len(dfofe.columns)):
      lowwofe.append(min(dfofe.iloc[:,an]))
      highhofe.append(max(dfofe.iloc[:,an]))
    rangparamofe = len(dfofel)
    radarofe = Radar(dfofel, lowwofe, highhofe,
                  # whether to round any of the labels to integers instead of decimal places
                  round_int=[False]*rangparamofe,
                  num_rings=4,  # the number of concentric circles (excluding center circle)
                  # if the ring_width is more than the center_circle_radius then
                  # the center circle radius will be wider than the width of the concentric circles
                  ring_width=1, center_circle_radius=1)

    colorradar1 = "#FF0046"
    colorradar2 = "#005CBE"
    alpharradar1 = 0.5
    alpharradar2 = 0.5
    st.write(valuessofe)
    metricplayerbox01, metricplayerbox02, metricplayerbox03 = st.columns(3)
    #with metricplayerbox01:
        #Team_Lst = df['Team'].drop_duplicates().tolist()
        #Metric_Lst = columnsevents
        #MetricSel = st.selectbox("Choose metric:", Metric_Lst)
    
    space0, space1, space2 = st.columns((0.6, 0.6, 0.6))
    with space0:

        fig, ax = radarofe.setup_axis()  # format axis as a radar
        fig.set_facecolor('#050E1E')
        fig.set_dpi(600)
        st.markdown('<h1 style="font-size: 25px;">OFFENSIVE</h1>', unsafe_allow_html=True)
    
        rings_inner = radarofe.draw_circles(ax=ax, facecolor=(1,1,1,0), edgecolor='#222229')  # draw circles
        radar_output = radarofe.draw_radar_compare(valuessofe, valuessofe2, ax=ax,
                                                kwargs_radar={'facecolor': colorradar1, 'alpha' : alpharradar1},
                                                kwargs_compare={'facecolor': colorradar2, 'alpha' : alpharradar2},
                                                )  # draw the radar
        radar_poly, radar_poly2, vertices, vertices2 = radar_output
        # range_labels = radar.draw_range_labels(ax=ax, fontsize=18,
        #                                        fontproperties=prop)  # draw the range labels
        param_labels = radarofe.draw_param_labels(ax=ax, fontsize=15, color=(1,1,1,0.8),
                                               fontproperties=prop2)  # draw the param labels
    
        vert = vertices.tolist()
        dfver = pd.DataFrame(vert, columns=['X', 'Y'])
        uno = dfver['X'].tolist()
        dos = dfver['Y'].tolist()
    
        ax.scatter(vertices[:, 0], vertices[:, 1], c=colorradar1, edgecolors='#050E1E', s=120, alpha=alpharradar1, zorder=-1)
        ax.scatter(vertices2[:, 0], vertices2[:, 1], c=colorradar2, edgecolors='#050E1E', s=120, alpha=alpharradar2, zorder=-1)
    
        #st.write(lowwofe)
        #st.write(highhofe)
    
        st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=600, format="png")   
    with space1:
        st.markdown('<h1 style="font-size: 25px;">DEFENSIVE</h1>', unsafe_allow_html=True)
    with space2:
        st.markdown('<h1 style="font-size: 25px;">POSSESION</h1>', unsafe_allow_html=True)
    space3, space4, space5 = st.columns(3)
    with space3:
        st.markdown('<h1 style="font-size: 25px;">CREATION</h1>', unsafe_allow_html=True)
    with space4:
        st.markdown('<h1 style="font-size: 25px;">DISTRIBUTION</h1>', unsafe_allow_html=True)
    with space5:
        st.markdown('<h1 style="font-size: 25px;">SET PIECE</h1>', unsafe_allow_html=True)
    st.divider()
