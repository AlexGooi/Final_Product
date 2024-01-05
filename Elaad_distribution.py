import pandas as pd
import matplotlib.pyplot as plt
##import seaborn as sns 
import scipy.stats as stats
from scipy.stats import gamma
import numpy as np
import pickle

file_path1 = "CSV/open_transactions.csv"
file_path2 = "CSV/open_metervalues.csv"

df1 = pd.read_csv(file_path1, delimiter=';', decimal=',')
df2 = pd.read_csv(file_path2, delimiter=';', decimal=',')

# Data cleaning: Filter out rows from df1 where TotalEnergy is greater than 70
df1 = df1[df1['TotalEnergy'] <= 70]

# Convert columns to datetime
df1['UTCTransactionStart'] = pd.to_datetime(df1['UTCTransactionStart'], format='%d/%m/%Y %H:%M')
df1['UTCTransactionStop'] = pd.to_datetime(df1['UTCTransactionStop'], format='%d/%m/%Y %H:%M')

# Extract the hours and days for ALL data
df1['ArrivalHour'] = df1['UTCTransactionStart'].dt.hour
df1['DepartureHour'] = df1['UTCTransactionStop'].dt.hour
df1['ArrivalMinute'] = df1['UTCTransactionStart'].dt.hour * 60 + df1['UTCTransactionStart'].dt.minute
df1['DayOfWeek'] = df1['UTCTransactionStart'].dt.dayofweek

# Segregate df1 into weekdays and weekends
#df1_weekdays = df1[df1['DayOfWeek'].between(0, 4)]
#df1_weekends = df1[df1['DayOfWeek'].between(5, 6)]


##If data seggregated

# Calculate the average number of arrivals per hour for weekdays
#average_hourly_arrivals_weekdays = df1_weekdays.groupby('ArrivalHour').size().reset_index(name='AverageArrivals')
#average_hourly_arrivals_weekdays['AverageArrivals'] /= len(df1_weekdays['DayOfWeek'].unique())

# Average hourly departures for weekdays
#average_hourly_departures_weekdays = df1_weekdays.groupby('DepartureHour').size().reset_index(name='AverageDepartures')
#average_hourly_departures_weekdays['AverageDepartures'] /= len(df1_weekdays['DayOfWeek'].unique())

# Average hourly arrivals for weekends
#average_hourly_arrivals_weekends = df1_weekends.groupby('ArrivalHour').size().reset_index(name='AverageArrivals')
#average_hourly_arrivals_weekends['AverageArrivals'] /= len(df1_weekends['DayOfWeek'].unique())

# Average hourly departures for weekends
#average_hourly_departures_weekends = df1_weekends.groupby('DepartureHour').size().reset_index(name='AverageDepartures')
#average_hourly_departures_weekends['AverageDepartures'] /= len(df1_weekends['DayOfWeek'].unique())

## Take full df1
average_hourly_arrivals = df1.groupby('ArrivalHour').size().reset_index(name='AverageArrivals')
average_hourly_arrivals['AverageArrivals'] /= len(df1['DayOfWeek'].unique())

average_hourly_departures = df1.groupby('DepartureHour').size().reset_index(name='AverageDepartures')
average_hourly_departures['AverageDepartures'] /= len(df1['DayOfWeek'].unique())


# Function to plot the distribution
def plot_average_distribution(arrival_df1, departure_df1, title_suffix, colors=['#B9D531', '#EC008C']):
    combined_df1 = pd.merge(arrival_df1, departure_df1, left_on='ArrivalHour', right_on='DepartureHour', how='outer')
    combined_df1 = combined_df1.fillna(0)  # Fill NaN values with 0
    combined_df1.plot(x='ArrivalHour', y=['AverageArrivals', 'AverageDepartures'], kind='bar', figsize=(10, 5), color=colors)
    plt.title(f'Average Arrival and Departure Distribution over 24 Hours ({title_suffix})')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Count')
    plt.xticks(range(24))
    plt.tight_layout()

# Function to distributions
def calculate_distribution_params_at(df1): #Gamma distribution is best fit
    df1_sorted_at = df1.sort_values(by='UTCTransactionStart')
    df1_sorted_at['TimeDiff'] = df1_sorted_at['UTCTransactionStart'].diff().dt.total_seconds() / 60
    df1_sorted_at = df1_sorted_at.dropna(subset=['TimeDiff'])
    df1_sorted_at['TimeDiff'] = np.where(df1_sorted_at['TimeDiff'] <= 0, 0.001, df1_sorted_at['TimeDiff']) # zeros to super small
    params_gamma_at = stats.gamma.fit(df1_sorted_at['TimeDiff'], floc=0)
    params_expon_at = stats.expon.fit(df1_sorted_at['TimeDiff'], floc=0)
    params_lognorm_at = stats.lognorm.fit(df1_sorted_at['TimeDiff'], floc=0)
    return params_gamma_at, params_expon_at, params_lognorm_at, df1_sorted_at

def calculate_available_service_time_distribution(df1): #Lognorm distribution is best fit, only Gamma distribution is best fit before filtering.
    df1_sorted_ast = df1.sort_values(by='UTCTransactionStart') #sort by arrival time
    df1_sorted_ast['AvailableServiceTime'] = (df1_sorted_ast['UTCTransactionStop'] - df1_sorted_ast['UTCTransactionStart']).dt.total_seconds() / 60
    df1_sorted_ast = df1_sorted_ast.dropna(subset=['AvailableServiceTime'])
    df1_sorted_ast = df1_sorted_ast[df1_sorted_ast['AvailableServiceTime'] > 0]
    params_gamma_ast = stats.gamma.fit(df1_sorted_ast['AvailableServiceTime'], floc=0)
    params_expon_ast = stats.expon.fit(df1_sorted_ast['AvailableServiceTime'], floc=0)
    params_lognorm_ast = stats.lognorm.fit(df1_sorted_ast['AvailableServiceTime'], floc=0)

    return params_gamma_ast, params_expon_ast, params_lognorm_ast, df1_sorted_ast


def calculate_distribution_params_te(df1): #lognormal distribution is best fit
    df1_sorted_te = df1.sort_values(by='UTCTransactionStart') #TotalEnergy sorted on the arrivaltime of the EV
    params_gamma_te = stats.gamma.fit(df1_sorted_te['TotalEnergy'], floc=0)
    params_expon_te = stats.expon.fit(df1_sorted_te['TotalEnergy'], floc=0)
    params_lognorm_te = stats.lognorm.fit(df1_sorted_te['TotalEnergy'], floc=0)
    return params_gamma_te, params_expon_te, params_lognorm_te, df1_sorted_te

def calculate_statistical_metrics(data, params_gamma, params_expon, params_lognorm):
    ks_stat_gamma, p_value_gamma = stats.kstest(data, 'gamma', params_gamma)
    ks_stat_expon, p_value_expon = stats.kstest(data, 'expon', params_expon)
    ks_stat_lognorm, p_value_lognorm = stats.kstest(data, 'lognorm', params_lognorm)

    aic_gamma = calculate_aic_bic(data, 'gamma', params_gamma)
    aic_expon = calculate_aic_bic(data, 'expon', params_expon)
    aic_lognorm = calculate_aic_bic(data, 'lognorm', params_lognorm)

    bic_gamma = calculate_aic_bic(data, 'gamma', params_gamma, use_bic=True)
    bic_expon = calculate_aic_bic(data, 'expon', params_expon, use_bic=True)
    bic_lognorm = calculate_aic_bic(data, 'lognorm', params_lognorm, use_bic=True)

    return {
        'gamma': {'ks_stat': ks_stat_gamma, 'p_value': p_value_gamma, 'aic': aic_gamma, 'bic': bic_gamma},
        'expon': {'ks_stat': ks_stat_expon, 'p_value': p_value_expon, 'aic': aic_expon, 'bic': bic_expon},
        'lognorm': {'ks_stat': ks_stat_lognorm, 'p_value': p_value_lognorm, 'aic': aic_lognorm, 'bic': bic_lognorm}
    }


def calculate_aic_bic(data, dist_name, params, use_bic=False):
    k = len(params)
    llk = np.sum(getattr(stats, dist_name).logpdf(data, *params))
    return -2 * llk + k * (np.log(len(data)) if use_bic else 2)


#BEST FIT IS GAMMA FOR ARRIVAL TIME
params_gamma_at, params_expon_at, params_lognorm_at, df1_sorted_at = calculate_distribution_params_at(df1)
metrics_at = calculate_statistical_metrics(df1_sorted_at['TimeDiff'], params_gamma_at, params_expon_at, params_lognorm_at)

#INCONCLUSIVE: after filtering <=70 better fit lognorm, otherwise gamma
params_gamma_ast, params_expon_ast, params_lognorm_ast, df1_sorted_ast = calculate_available_service_time_distribution(df1)
metrics_ast = calculate_statistical_metrics(df1_sorted_ast['AvailableServiceTime'], params_gamma_ast, params_expon_ast, params_lognorm_ast)

#BEST FIT IS LOGNORMAL FOR TOTAL ENERGY
params_gamma_te, params_expon_te, params_lognorm_te, df1_sorted_te = calculate_distribution_params_te(df1)
metrics_te = calculate_statistical_metrics(df1_sorted_te['TotalEnergy'], params_gamma_te, params_expon_te, params_lognorm_te)

mean_arrival_time = gamma.mean(*params_gamma_at)


print("Gamma distribution parameters Arrival time:", params_gamma_at)
print("Lognorm distribution parameters Total Energt:", params_lognorm_te)
print("Gamma distribution parameters Available Service Time:", params_gamma_ast)
print("Metric Total Energy distributions", metrics_te)
print("Metric Available Service Time distributions", metrics_ast)
print("Metric Arrival Time distributions", metrics_at)
print("Mean Arrival Time (minutes):", mean_arrival_time)

print("Max AvailableServiceTime:", df1_sorted_ast['AvailableServiceTime'].max())



#save params_gamma LOOK! I AM A PICKLE
with open('params_gamma_at.pkl', 'wb') as f:
    pickle.dump(params_gamma_at, f)

with open('params_gamma_ast.pkl', 'wb') as f:
    pickle.dump(params_gamma_ast, f)

with open('params_lognorm_ast.pkl', 'wb') as f:
    pickle.dump(params_lognorm_ast, f)

with open('params_lognorm_te.pkl', 'wb') as f:
    pickle.dump(params_lognorm_te, f)

# # Generate points on the x axis suitable for the range of your data
# x_gamma_at = np.linspace(start=0, stop=df1_sorted_at['TimeDiff'].max(), num=10000)
# #x_expon_at = np.linspace(start=0, stop=df1_sorted_at['AvailableServiceTime'].max(), num=10000)
# #x_lognorm_at = np.linspace(start=0, stop=df1_sorted_at['TotalEnergy'].max(), num=10000)


# # Plot the histogram of the empirical data
# plt.figure(figsize=(15, 6))
# bin_size = 1  # The bin size can be adjusted to increase of decrease granularity
# bins = int((df1_sorted_at['TimeDiff'].max() - df1_sorted_at['TimeDiff'].min()) / bin_size)
# sns.histplot(df1_sorted_at['TimeDiff'], bins=bins, kde=False, stat='density', label='Empirical Data', color='#FBD5EC')

# # Plot the PDF of the fitted gamma distribution
# pdf_gamma_at = stats.gamma.pdf(x_gamma_at, *params_gamma_at)
# plt.plot(x_gamma_at, pdf_gamma_at, label='Fitted Gamma Distribution', color='#B9D531')

# # Plot the PDF of the fitted exponential distribution
# pdf_expon_at = stats.expon.pdf(x_gamma_at, *params_expon_at)
# plt.plot(x_gamma_at, pdf_expon_at, label='Fitted Exponential Distribution', color='#EC008C')

# # Plot the PDF of the fitted exponential distribution
# pdf_lognorm_at = stats.lognorm.pdf(x_gamma_at, *params_lognorm_at)
# plt.plot(x_gamma_at, pdf_lognorm_at, label='Fitted Lognormal Distribution', color='#696A6C')


# # Finalize the plot (same as before)
# plt.xlabel('Time Difference Between Arrivals (Minutes)')
# plt.ylabel('Density')
# plt.title('Average Daily Frequency of Time Differences Between Arrivals with Fitted Distributions')
# plt.xlim(0, df1_sorted_at['TimeDiff'].quantile(0.95))
# plt.legend()
# plt.show()