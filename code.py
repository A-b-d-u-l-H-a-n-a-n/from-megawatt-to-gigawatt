import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import wbgapi as wb
import warnings
warnings.filterwarnings('ignore')

# =========================Data Gathering============================

def take_data(indicator): 
    data = wb.data.DataFrame(indicator)
    df = pd.DataFrame(data)
    df_t = df.T 
    return df, df_t

electric, electric_t = take_data('EG.USE.ELEC.KH.PC')
electric_access, electric_access_t = take_data('EG.ELC.ACCS.ZS')
ep_oil, ep_oil_t = take_data('EG.ELC.PETR.ZS')
ep_nuclear, ep_nuclear_t = take_data('EG.ELC.NUCL.ZS')
ep_natural_gas, ep_natural_gas_t = take_data('EG.ELC.NGAS.ZS')
ep_coal, ep_coal_t = take_data('EG.ELC.COAL.ZS')
renew_energy, renew_energy_t = take_data('EG.FEC.RNEW.ZS')
gas_emission, gas_emission_t = take_data('EN.ATM.GHGT.KT.CE')
urban_p, urban_p_t = take_data('SP.URB.TOTL')
forest, forest_t = take_data('AG.LND.FRST.K2')


#=========================Data Transformation=============================
def data_transform(df):
    renamed_df = df.copy()
    renamed_df.columns.names = ['Country Code']
    renamed_df.index = np.arange(1960,2022)
    renamed_df.index.names = ['Year']
    return renamed_df

transform_electric = data_transform(electric_t)
transform_electric_access = data_transform(electric_access_t)
transform_ep_oil = data_transform(ep_oil_t)
transform_ep_nuclear = data_transform(ep_nuclear_t)
transform_ep_natural_gas = data_transform(ep_natural_gas_t)
transform_ep_coal = data_transform(ep_coal_t)
transform_renew_energy = data_transform(renew_energy_t)
transform_gas_emission = data_transform(gas_emission_t)
transform_urban_p = data_transform(urban_p_t)
transform_forest = data_transform(forest_t)

#============================Data Cleaning==============================
def handling_null(df):
    cleaned_df = df.copy()
    cleaned_df.dropna(how='all', axis = 1, inplace = True)
    cleaned_df.dropna(how='all', axis = 0, inplace = True)
    cleaned_df.fillna(method='bfill', inplace = True)
    cleaned_df = cleaned_df.loc[1971:2014,:]
    return cleaned_df

cleaned_electric = handling_null(transform_electric)
cleaned_electric_access = handling_null(transform_electric_access)
cleaned_ep_oil = handling_null(transform_ep_oil)
cleaned_ep_nuclear = handling_null(transform_ep_nuclear)
cleaned_ep_natural_gas = handling_null(transform_ep_natural_gas)
cleaned_ep_coal= handling_null(transform_ep_coal)
cleaned_renew_energy = handling_null(transform_renew_energy)
cleaned_gas_emission = handling_null(transform_gas_emission)
cleaned_urban_p = handling_null(transform_urban_p)
cleaned_forest = handling_null(transform_forest)

# Aggreagating global data
def get_global_data(df, name, metric = '%'):
    year_list = df.index
    value_by_time = [df.loc[year,:].mean() for year in year_list]
    
    if metric == 'kWh':
        value_by_time = np.array([df.loc[year,:].sum() for year in year_list]) / 1000
    elif metric == 'kt':
        value_by_time = np.array([df.loc[year,:].sum() for year in year_list]) / 1e+6
        
    global_data = pd.DataFrame({
        'Year': year_list, 
        name: value_by_time
    }
    )
    global_data.set_index('Year', inplace= True)
    
    return global_data

# Calculating avg global data
def get_avg_per_year(df):
    differences = []
    start_year = df.index[0]
    end_year = df.index[-1]
    
    for i in range(1, len(df)):
        differences.append(abs(df.iloc[i-1] - df.iloc[i]))

    difference_avg = sum(differences) / len(differences)
    return round(difference_avg, 3)

# Aggregating country data
def agg_data(country):
    country_electric = cleaned_electric.loc[1971:2014, country]
    country_ep_oil = cleaned_ep_oil.loc[1971:2014, country]
    country_ep_nuclear = cleaned_ep_nuclear.loc[1971:2014, country]
    country_ep_natural_gas = cleaned_ep_natural_gas.loc[1971:2014, country]
    country_ep_coal = cleaned_ep_coal.loc[1971:2014, country]
    country_electric_access = cleaned_electric_access.loc[1971:2014, country]
    country_renew_energy = cleaned_renew_energy.loc[1971:2014, country]
    country_gas_emission = cleaned_gas_emission.loc[1971:2014, country]
    country_urban_p = cleaned_urban_p.loc[1971:2014, country]
    country_forest = cleaned_forest.loc[1971:2014, country]
    
    concatenated_data = pd.concat([
        country_electric, country_ep_oil, 
        country_ep_nuclear, country_ep_natural_gas,
        country_ep_coal, country_electric_access,
        country_renew_energy, country_gas_emission, 
        country_urban_p, country_forest
    ], axis=1
    )
    
    concatenated_data.columns = [
        'Electric Power (kWh)', 'EP Oil (%)', 
        'EP Nuclear (%)', 'EP Natural Gas (%)', 
        'EP Coal (%)', 'Electric Access (%)',
        'Renewable Energy (%)', 'Gas Emission (kt)', 
        'Urban Population', 'Forest Area (sq. km)'
    ]
    
    return concatenated_data

# Generate correlation chart
def add_corr_chart(fig, x, y, xlabel, ylabel, color):
    fig.add_trace(
        go.Scatter(
            x = x, 
            y = y, mode='markers',
            name = 'Urban Population',
            line= dict(
                width = 3, 
                color = color
            ), marker=dict(size=15),
            line_shape = 'spline'
        ),
    )
    fig.update_layout(
        xaxis_title = xlabel, yaxis_title = ylabel,
        showlegend = False,
        plot_bgcolor = 'white',
        xaxis = dict(
            showgrid = True,
            gridcolor ='rgb(204, 204, 204)',
            showticklabels = True,
            linecolor = 'rgb(204, 204, 204)',
            linewidth = 2,
            ticks = 'outside',
            tickfont = dict(
                family = 'Arial',
                size = 15,
                color = 'rgb(82, 82, 82)',
            ),
        ),
        yaxis = dict(
            showgrid = True,
            gridcolor ='rgb(204, 204, 204)',
            showticklabels = True,
            linecolor = 'rgb(204, 204, 204)',
            ticks = 'outside', 
            tickfont = dict(
                family = 'Arial',
                size = 15,
                color = 'rgb(82, 82, 82)',
            ),
        ),

    )
    return fig

# Adding timeseries line
def add_chart(fig, data, indicator, label, color):
    start_year = data.index[0]
    end_year = data.index[-1]
    
    # Add line
    fig.add_trace(
        go.Scatter(
            x = data.index, 
            y = data[indicator],
            name = label,
            line= dict(
                width = 3, 
                color = color
            ), 
            line_shape = 'spline'
        ),
    )
    
    # Add start point
    fig.add_trace(
        go.Scatter(
            x = np.array(start_year),
            y = np.array(data.loc[start_year, indicator]),
            mode = 'markers',
            marker = dict(
                size = 10, 
                color = color
            ),
        ),
    )
    
    # Add end point
    fig.add_trace(
        go.Scatter(
            x = np.array(end_year + 1.5), 
            y = np.array(data.loc[end_year, indicator]),
            mode = 'markers',
            marker = dict(
                size = 10, 
                color = color
            ),
        ),
    )
    return fig

# Adding timeseries line (Extension)
def eps_line(fig, global_data, source_name, label, color):
    y_data = global_data[source_name].values

    fig = add_chart(fig, global_data, source_name, label, color)
    annot(y_data, label)
    return fig

# Customize plot template
def fig_template(fig):
    fig.update_layout(
        width = 700,
        xaxis = dict(
            showline = True,
            showgrid = False,
            showticklabels = True,
            linecolor = 'rgb(204, 204, 204)',
            linewidth = 2,
            ticks = 'outside',
            tickfont = dict(
                family = 'Arial',
                size = 12,
                color = 'rgb(82, 82, 82)',
            ),
        ),
        yaxis = dict(
            showgrid = False,
            zeroline = False,
            showline = False,
            showticklabels = False,
        ),
        autosize = False,
        margin = dict(
            autoexpand=False,
            l = 105,
            r = 50,
            t = 110,
        ),
        showlegend = False,
        plot_bgcolor = 'white'
    )
    
    return fig

# Resetting annotation
def reset_annot():
    global annotations
    annotations = []
    
# Function for annotating plot 
def annot(y_data, labels, metric = '%'):
    if metric == 'kWh':
        metric = 'mWh'
    elif metric == 'kt':
        metric = 'gt'
    elif metric != '%' and metric != 'kWh':
        metric = 'B'
    # Add label on left_side of the plot
    annotations.append(dict(
        xref = 'paper', 
        x = 0.05, 
        y = y_data[0],
        xanchor = 'right', yanchor = 'middle',
        text = labels +' {:.0f}{}'.format(y_data[0], metric),
        font = dict(
            family = 'Arial',
            size = 16
        ),
        showarrow = False
        
    ),
    )
    # Add label on right_side of the plot
    annotations.append(dict(
        xref = 'paper', 
        x = 0.95, 
        y = y_data[-1],
        xanchor = 'left', yanchor = 'middle',
        text = '{:.0f}{}'.format(y_data[-1], metric),
        font = dict(
            family = 'Arial',
            size = 16
        ),
        showarrow = False
    ),
    )
    
# Adding title in plot
def add_title(title, size = 30, position = 0):
    annotations.append(dict(
        xref = 'paper', yref = 'paper', 
        x = position, y = 1.05,
        xanchor = 'left', yanchor='bottom',
        text = title,
        font = dict(
            family = 'Arial',
            size = size,
            color = 'rgb(37,37,37)'
        ),
        showarrow = False
    ),
    )

# Adding text in plot 
def add_text(text):
    annotations.append(dict(
        xref = 'paper', yref = 'paper', 
        x = 0.5, y = -0.1,
        xanchor = 'center', yanchor = 'top',
        text = text,
        font = dict(
            family = 'Arial',
            size = 12,
            color = 'rgb(150,150,150)'
        ),
        showarrow = False
    ),
    )

# Creating correlation table
def corr_table(*list_data, start_year):
    df = []
    for data in list_data:
        df.append(data.loc[start_year:])
        
    corr_table = pd.concat(df, axis = 1).corr()
    return corr_table

# Generate heatmap
def make_heatmap(df,title):
    heatmap = px.imshow(
        df, text_auto = True, 
        color_continuous_scale = 'RdYlGn', 
        title = '<b>{}</b>'.format(title)
    )

    heatmap.update_layout(
        width = 550, title_x = 0.55,
        xaxis = dict(tickangle = 45),
        font = dict(
            family = 'Arial',
            size = 14,
            color = 'black'
        )
    )
    return heatmap
#==============================Figure. 1=============================================
# Create input for figure
global_electric = get_global_data(cleaned_electric, 'Electric Power (kWh)','kWh') 
y_data = global_electric['Electric Power (kWh)'].values
annual_decrease = get_avg_per_year(global_electric)[0]

# Creating plot
fig = go.Figure()
fig = fig_template(fig)
fig = add_chart(fig, global_electric, 'Electric Power (kWh)', 
                'Electric Power Consumption', '#FECB52')

# Customizing plot
reset_annot()
annot(y_data, 'Global', 'kWh')
add_title('Global Electric Power Consumption (1971 - 2014)(mWh)', 22)
text = 'Average annual increase: {}mWh'.format(annual_decrease)
add_text(text)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()
#===========================================================================

#==============================Figure.2 /========================================
# Create input for figure
global_gas_emission = get_global_data(cleaned_gas_emission, 'Gas Emission (kt)','kt') 
y_data = global_gas_emission['Gas Emission (kt)'].values
annual_decrease = get_avg_per_year(global_gas_emission)[0]

# Creating plot
fig = go.Figure()
fig = fig_template(fig)
fig = add_chart(fig, global_gas_emission, 'Gas Emission (kt)', 
                'Gas Emission', 'red')

# Customizing plot
reset_annot()
annot(y_data, 'Global', 'kt')
add_title('Global Gas Emission (1971 - 2014)(gt)')
text = 'Average annual increase: {}gt'.format(annual_decrease)
add_text(text)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()
#===============================================================

#=======================Figure. 3===========================================
# Create input for figure
global_oil = get_global_data(cleaned_ep_oil, 'EP Oil (%)')
global_nuclear = get_global_data(cleaned_ep_nuclear, 'EP Nuclear (%)')
global_natural_gas = get_global_data(cleaned_ep_natural_gas, 'EP Natural Gas (%)')
global_coal = get_global_data(cleaned_ep_coal, 'EP Coal (%)')

# Make a correlation matrix
matrix = corr_table(global_gas_emission,global_oil,
                    global_nuclear,global_natural_gas,global_coal,
                    start_year = 1971)

# Generate heatmap from correlation matrix
heatmap = make_heatmap(matrix,'Electric Source Production vs Gas Emission')
plot(heatmap, auto_open = True)
#==================================================================

#===========================Figure. 4=========================================
# Create input for figure
global_oil = global_oil.loc[1971:,]
global_nuclear = global_nuclear.loc[1971:,]
global_natural_gas = global_natural_gas.loc[1971:,]
global_coal = global_coal.loc[1971:,]

# Create plot
fig = go.Figure()
fig = fig_template(fig)
reset_annot()

# Adding chart to existing plot
fig = eps_line(fig, global_oil, 'EP Oil (%)','Oil', 'gray')
fig = eps_line(fig, global_nuclear, 'EP Nuclear (%)','Nuclear', 'gray')
fig = eps_line(fig, global_natural_gas, 'EP Natural Gas (%)','Natural Gas', 'red')
fig = eps_line(fig, global_coal, 'EP Coal (%)','Coal', 'gray')

# Adding title
add_title('Global Electric Production Sources (1971 - 2014)(%)', 24)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()
#========================================================

#=======================Figure. 5=========================================
# Create input for figure
global_urban_p = get_global_data(cleaned_urban_p, 'Urban Population', 'Billion')
global_electric_access = get_global_data(cleaned_electric_access, 'Electric Access (%)')
global_forest = get_global_data(cleaned_forest, 'Forest Area (sq. km)')

x = global_urban_p['Urban Population']
y = global_gas_emission['Gas Emission (kt)'].values

# Create plot
fig = go.Figure()
add_corr_chart(fig, x, y, 'Urban Population (Million)', 'Gas Emission (gt)', 'red')

# Generate plot
reset_annot()
add_title('Urban Population vs Gas Emission',position =0.14)
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()
#---------------------Figure. 6----------------------------------
# Make a correlation matrix
matrix = corr_table(global_urban_p,global_electric_access,global_forest,
                    start_year = 2000)

# Generate plot
heatmap = make_heatmap(matrix,'Urban Population Relationship')
plot(heatmap, auto_open = True)
#==========================================================

#=======================Figure. 7=====================================
# Create input for figure
denmark = agg_data('DNK')
denmark_renew_energy = denmark.loc[1990:, 'Renewable Energy (%)'].to_frame()
y_data = denmark_renew_energy['Renewable Energy (%)'].values

# Create plot
fig = go.Figure()
fig = fig_template(fig)
fig = add_chart(fig, denmark_renew_energy, 'Renewable Energy (%)', 
                'Renewable Energy', 'yellowgreen')

# Customizing plot
reset_annot()
annot(y_data, 'Denmark')
add_title('Denmark Renewable Energy Consumption (1990 - 2014)(%)', 20)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()
# ==================================

#=================Figure. 8================================
# Create input for figure
y = denmark.loc[1990:,'Gas Emission (kt)'].values
x = denmark.loc[1990:,'Renewable Energy (%)'].values

# Create plot
fig = go.Figure()
add_corr_chart(fig, x, y, 'Renewable Energy (%)', 'Gas Emission (kt)', 'yellowgreen')

# Adding plot
reset_annot()
add_title('Renewable Energy vs Gas Emission', position=0.14)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()
#================================================

#===============Figure. 9================================
# Create input for figure
denmark_renew_energy = denmark[['Renewable Energy (%)']]
denmark_electric = denmark[['Electric Power (kWh)']]
denmark_gas_emission = denmark[['Gas Emission (kt)']]

# Make a correlation matrix
matrix = corr_table(denmark_renew_energy, denmark_electric, denmark_gas_emission,
                    start_year = 1990)

# Generate heatmap from correlation matrix
heatmap = make_heatmap(matrix,'Renewable Energy Contribution in Denmark')
plot(heatmap, auto_open = True)
#=============================================
