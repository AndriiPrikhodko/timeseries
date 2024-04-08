# %%
from dateutil.parser import parse
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import requests
import io
import altair as alt

# %%
# plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
rSession = requests.Session()
res = rSession.get('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
# %%
# Import as Dataframe
df = pd.read_csv(io.StringIO(res.text), sep=',', engine='python', parse_dates=['date'], index_col='date')
values = np.array(df['value'])
sigma = np.std(values)
mean = np.mean(values)
# Sewall Wright test 3 sigma test

SW_list = list(filter(lambda x: (mean - 3 * sigma > x or x > mean + 3 * sigma), df['value']))
print(SW_list)

# Draw Plot
# def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
#     plt.figure(figsize=(16,5), dpi=dpi)
#     plt.plot(x, y, color='tab:red')
#     plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
#     plt.show()

# plot_df(df, x=df.index, y=df.value, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')

# Additive Decomposition
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')
# %%
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
# %%
# Multiplicative Decomposition
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
result_mul.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
# %%
# Get trend
trend_add = result_add.trend
trend_mul = result_mul.trend
trend_add.plot()
trend_mul.plot()
# %%
# Seasonal component
seasonal_add = result_add.seasonal
seasonal_add.plot()
# %%
# Model error
resid_add = result_add.resid
mod_add_error = np.sum(np.square(resid_add))
print(np.sqrt(mod_add_error))
# %%
# Check error is white noise N(0,1)
ksts = stats.kstest(resid_add,stats.norm.cdf)
if (ksts.pvalue > 0.05) :
    print('We accept null hypothesis that a model errors are distributed according to the standard normal with confidence level 95%')
# %%
# Seasonal period
seasonal_add = result_add.seasonal
print(seasonal_add)
# %%
# The unobserved components model
# Unrestricted model, using string specification
unrestricted_model = {
    'level': 'local linear trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
}

# Unrestricted model, setting components directly
# This is an equivalent, but less convenient, way to specify a
# local linear trend model with a stochastic damped cycle:
# unrestricted_model = {
#     'irregular': True, 'level': True, 'stochastic_level': True, 'trend': True, 'stochastic_trend': True,
#     'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
# }
output_mod = sm.tsa.UnobservedComponents(seasonal_add, **unrestricted_model)
output_res = output_mod.fit(method='powell', disp=False)
print(output_res.summary())
(sigma_irregular, sigma_level, sigma_trend,
         sigma_cycle, frequency_cycle, damping_cycle) = output_res.params

period = (2 * np.pi) / frequency_cycle
true_period = 12
jan = seasonal_add[1::true_period]
print(jan)
# %%
# Building plot
output_res.plot_components(legend_loc='lower right', figsize=(15, 9))
# %%
# next seasonal index
season_idx_next = seasonal_add[len(seasonal_add.index) - true_period]
print(season_idx_next)

# %%
# Build trend log regression
# dataset_trend = sm.datasets.get_rdataset("Duncan", "carData")
Y = np.log(trend_add.values)
X = trend_add.reset_index().index
X = sm.add_constant(X)
model = sm.OLS(Y,X)
# Fit and summarize OLS model
regress_mod = model.fit()
# regress_mod = sm.ols(formula="A ~ B + C", data=df).fit()
print(regress_mod.summary())
# %%
# Plot regression mod and log(Y)
Y_mod = regress_mod.fittedvalues
Idx_mod = trend_add.index
df_mod = pd.DataFrame(data = Y_mod, index = Idx_mod)
df_log = pd.DataFrame(data = Y, index = Idx_mod)
plt.plot(df_mod, color='g')
plt.plot(df_log, color='r')
plt.show()

# %%
# Revert to actual values
Y_mod_exp = np.exp(Y_mod)
Y_exp = trend_add.values
df_mod_exp = pd.DataFrame(data = Y_mod_exp, index = Idx_mod)
df_exp = pd.DataFrame(data = Y_exp, index = Idx_mod)
plt.plot(df_mod_exp, color='g')
plt.plot(df_exp, color='r')
plt.show()

# %%
# get model constants
(intercept, slope) = regress_mod.params
exp_intercept = np.exp(intercept)

# %%
# trend predictions linear
predictor=10
arg = np.arange(df.size+predictor, dtype=int)
regression_func = lambda x: x * slope + intercept
regression_vfunc = np.vectorize(regression_func)
Y_mod_forecast_lin = regression_vfunc(arg)
df_mod_lin = pd.DataFrame(data = Y_mod_forecast_lin, index = arg)
# how to apply original index labels
plt.plot(df_mod_lin, color='g')
plt.show()
# %%
# trend prediction exp
regression_exp_func = lambda x: np.exp(x * slope) * exp_intercept
regression_exp_vfunc = np.vectorize(regression_exp_func)
Y_mod_exp_forecast = regression_exp_vfunc(arg)
df_mod_exp = pd.DataFrame(data = Y_mod_exp_forecast, index = arg)
plt.plot(df_mod_exp, color='g')
plt.show()
# %%
# prediction with seasonal component
seasonal_period = seasonal_add.values
seasonal_arg = np.concatenate([seasonal_add, seasonal_period[1:predictor+1]])
Y_mod_seasonal_forecast = np.add(seasonal_arg, Y_mod_exp_forecast)
df_mod_exp_seasonal = pd.DataFrame(data = Y_mod_seasonal_forecast, index = arg)
plt.plot(df_mod_exp_seasonal, color='g')
df_indx = pd.DataFrame(data=df.values, index=np.arange(df.size, dtype=int))
plt.plot(df_indx, color='r')
plt.show()
# %%
# look at ARCH model 2003 Nobel prise
# heteroskedasticity in econometric models, the best test is the White test.
manual_factor = np.log(np.multiply(arg, slope)+intercept) + np.exp((arg-102)/102) * 0.2
# + np.log((arg/204)+1)
seasonal_arg_heter = np.multiply(seasonal_arg, manual_factor)
Y_mod_seasonal_forecast_heter = np.add(seasonal_arg_heter, Y_mod_exp_forecast)
df_mod_exp_seasonal_heter = pd.DataFrame(data = Y_mod_seasonal_forecast_heter, index = arg)

df_indx = pd.DataFrame(data=df.values, index=np.arange(df.size, dtype=int))

df_mod_exp_seasonal_heter.columns = ['value']
df_indx.columns = ['value']

heter_chart = alt.Chart(df_mod_exp_seasonal_heter.reset_index()).encode(
        x = 'index:T',
        y = 'value:Q'
    )

df_chart = alt.Chart(df_indx.reset_index()).encode(
        x = 'index:T',
        y = 'value:Q'
    )

alt.layer(
    heter_chart.mark_line(color='green'),
    df_chart.mark_line(color='red'),
).interactive()
# %%
# Error estimate
resid_heter = np.add(df, -df_mod_exp_seasonal_heter[0:204])
mod_error = np.sum(np.square(resid_heter))
print(np.sqrt(mod_error))
diff_error = abs(np.sqrt(mod_error) - np.sqrt(mod_add_error))
print(1-np.sqrt(mod_error)/np.sqrt(mod_add_error))
# %%
# align names and X scale
df_mod_exp_seasonal_heter.columns = ['value']
df_mod_exp_seasonal.columns = ['value']
df.index = np.arange(df.size, dtype=int)

# %%
# model_type = alt.binding_select(options=['forecast','original'], name='models')
# selection = alt.selection_single(fields=['value'], bind=model_type)
# color = alt.condition(selection,
#                     alt.Color('value:O',legend=None),
#                     alt.value('white'))

prediction_chart = alt.Chart(df_mod_exp_seasonal_heter.reset_index()).encode(
        x = 'index:T',
        y = 'value:Q'
    )

prediction_chart_stsmd = alt.Chart(df_mod_exp_seasonal.reset_index()).encode(
        x = 'index:T',
        y = 'value:Q'
    )

default_chart = alt.Chart(df.reset_index()).encode(
        x = 'index:T',
        y = 'value:Q'
    )

alt.layer(
    prediction_chart.mark_line(color='green'),
    prediction_chart_stsmd.mark_line(),
    default_chart.mark_line(color='red')
).interactive()
# %%
# calculating return
# log provides negative values
# in case previous value is highter then following one
df['return'] = np.log(df['value'] /
                         df['value'].shift(1))
# %%
# adding direction to df
# 1 for positive return
# 0 for negative return
df['direction'] = np.where(df['return'] > 0, 1, 0)
# %%
# composing feed
cols = []
# lags define input range into NN
lags = 5
for lag in range(1, lags + 1): # <5>
    col = f'lag_{lag}'
    df[col] = df['return'].shift(lag) # <6>
    cols.append(col)
df.dropna(inplace=True)
# %%
# Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# %%
# set random seed
tf.random.set_seed(100)
# %%
# normalization of data
mx, sigma = np.nanmean(df.values), np.nanstd(df.values)
df_normalized = df.apply(lambda x: (x - mx) / sigma, axis=1)
# test normilized
mx_norm, sigma_norm = np.nanmean(df_normalized.values), np.nanstd(df_normalized.values)
# %%
# plot normailized data
alt.Chart(df_normalized.reset_index()).mark_line().encode(
        x = 'index:T',
        y = 'value:Q'
    ).interactive()

# %%
# specifying training and prediction data
# defining cutoff
cutoff = 100
training_data = df[df.index < cutoff].copy()
test_data = df[df.index >= cutoff].copy()
#  %%
# normilized data
mu, std = training_data.mean(), training_data.std()
test_data_ = (test_data - mu) / std
training_data_ = (training_data - mu) / std
# %%
# configure NN
alpha = 0.001
opt = keras.optimizers.Adam(learning_rate=alpha)

model = keras.Sequential()

model.add(layers.Dense(64, activation='relu',
        input_shape=(lags,)))
model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dense(1, activation="sigmoid"))
# %%
# build NN
# model.compile(
#     optimizer=opt,  # Optimizer
#     # Loss function to minimize
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     # List of metrics to monitor
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
# %%
# train NN
model.fit(training_data_[cols],
          training_data['direction'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False)
# %%
# results
res = pd.DataFrame(model.history.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--');
# %%
# eval test data
model.evaluate(training_data_[cols], training_data['direction'])
# %%
# predict based on trained data
predict = model.predict(training_data_[cols])
predict[:30].flatten()
# %%
# classify prediction direction
training_data['prediction'] = np.where(predict > 0, 1, -1)