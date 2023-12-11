import pandas as pd
import matplotlib.pyplot as plt
##import seaborn as sns 
import scipy.stats as stats
from scipy.stats import gamma
import numpy as np
import pickle

file_path1 = "./CSV/open_transactions.csv"
file_path2 = "./CSV/open_transactions.csv"

df1 = pd.read_csv(file_path1, delimiter=';', decimal=',')
df2 = pd.read_csv(file_path2, delimiter=';', decimal=',')

# Convert the columns to datetime
df1['UTCTransactionStart'] = pd.to_datetime(df1['UTCTransactionStart'], format='%d/%m/%Y %H:%M')
df1['UTCTransactionStop'] = pd.to_datetime(df1['UTCTransactionStop'], format='%d/%m/%Y %H:%M')

# Extract the hours and days
df1['ArrivalHour'] = df1['UTCTransactionStart'].dt.hour
df1['DepartureHour'] = df1['UTCTransactionStop'].dt.hour
df1['DayOfWeek'] = df1['UTCTransactionStart'].dt.dayofweek
df1['ArrivalMinute'] = df1['UTCTransactionStart'].dt.hour * 60 + df1['UTCTransactionStart'].dt.minute

# Segregate df1 into weekdays and weekends
df_weekdays = df1[df1['DayOfWeek'].between(0, 4)]
df_weekends = df1[df1['DayOfWeek'].between(5, 6)]

# Separate arrival and departure dataframes for weekdays and weekends
arrival_weekdays = df_weekdays[['ArrivalHour']]
departure_weekdays = df_weekdays[['DepartureHour']]
arrival_weekends = df_weekends[['ArrivalHour']]
departure_weekends = df_weekends[['DepartureHour']]

# Calculate the average number of arrivals per hour for weekdays
average_hourly_arrivals_weekdays = df_weekdays.groupby('ArrivalHour').size().reset_index(name='AverageArrivals')
average_hourly_arrivals_weekdays['AverageArrivals'] /= len(df_weekdays['DayOfWeek'].unique())

# Average hourly departures for weekdays
average_hourly_departures_weekdays = df_weekdays.groupby('DepartureHour').size().reset_index(name='AverageDepartures')
average_hourly_departures_weekdays['AverageDepartures'] /= len(df_weekdays['DayOfWeek'].unique())

# Average hourly arrivals for weekends
average_hourly_arrivals_weekends = df_weekends.groupby('ArrivalHour').size().reset_index(name='AverageArrivals')
average_hourly_arrivals_weekends['AverageArrivals'] /= len(df_weekends['DayOfWeek'].unique())

# Average hourly departures for weekends
average_hourly_departures_weekends = df_weekends.groupby('DepartureHour').size().reset_index(name='AverageDepartures')
average_hourly_departures_weekends['AverageDepartures'] /= len(df_weekends['DayOfWeek'].unique())

# Function to plot the distribution
def plot_average_distribution(arrival_df, departure_df, title_suffix, colors=['#B9D531', '#EC008C']):
    # Merge the dataframes on the hour
    combined_df = pd.merge(arrival_df, departure_df, left_on='ArrivalHour', right_on='DepartureHour', how='outer')
    combined_df = combined_df.fillna(0)  # Fill NaN values with 0
    combined_df.plot(x='ArrivalHour', y=['AverageArrivals', 'AverageDepartures'], kind='bar', figsize=(10, 5), color=colors)
    plt.title(f'Average Arrival and Departure Distribution over 24 Hours ({title_suffix})')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Count')
    plt.xticks(range(24))
    plt.tight_layout()

# Function to distributions
def calculate_distribution_params(df):
    df_sorted = df.sort_values(by='UTCTransactionStart')
    df_sorted['TimeDiff'] = df_sorted['UTCTransactionStart'].diff().dt.total_seconds() / 60
    df_sorted = df_sorted.dropna(subset=['TimeDiff'])
    df_sorted['TimeDiff'] = np.where(df_sorted['TimeDiff'] <= 0, 0.001, df_sorted['TimeDiff'])
    params_gamma = stats.gamma.fit(df_sorted['TimeDiff'], floc=0)
    params_expon = stats.expon.fit(df_sorted['TimeDiff'], floc=0)
    return params_gamma, params_expon, df_sorted

# Plotting for weekdays and weekends
#plot_average_distribution(average_hourly_arrivals_weekdays, average_hourly_departures_weekdays, 'Weekdays', colors=['#B9D531', '#EC008C'])
#plot_average_distribution(average_hourly_arrivals_weekends, average_hourly_departures_weekends, 'Weekends', colors=['#B9D531', '#EC008C'])
#plt.show()

params_gamma, params_expon, df1_sorted = calculate_distribution_params(df1)

#save params_gamma LOOK! I AM A PICKLE
with open('params_gamma.pkl', 'wb') as f:
    pickle.dump(params_gamma, f)

# Generate points on the x axis suitable for the range of your data
x_gamma = np.linspace(start=0, stop=df1_sorted['TimeDiff'].max(), num=10000)
x_expon = np.linspace(start=0, stop=df1_sorted['TimeDiff'].max(), num=10000)


# Plot the histogram of the empirical data
#plt.figure(figsize=(15, 6))
##sns.histplot(df1_sorted['TimeDiff'], bins=bins, kde=False, stat='density', label='Empirical Data', color='#FBD5EC')

# Plot the PDF of the fitted gamma distribution
#pdf_gamma = stats.gamma.pdf(x_gamma, *params_gamma)
#plt.plot(x_gamma, pdf_gamma, label='Fitted Gamma Distribution', color='#B9D531')

# Plot the PDF of the fitted exponential distribution
#pdf_expon = stats.expon.pdf(x_expon, *params_expon)
#plt.plot(x_expon, pdf_expon, label='Fitted Exponential Distribution', color='#EC008C')


# Finalize the plot (same as before)
#plt.xlabel('Time Difference Between Arrivals (Minutes)')
#plt.ylabel('Density')
#plt.title('Average Daily Frequency of Time Differences Between Arrivals with Fitted Distributions')
#plt.xlim(0, df1_sorted['TimeDiff'].quantile(0.95))
#plt.legend()
#plt.show()