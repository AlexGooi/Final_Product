import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
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

# Define time segments
morning_start, morning_end = 6, 10
afternoon_start, afternoon_end = 10, 16
early_evening_start, early_evening_end = 16, 20
late_evening_start, late_evening_end = 20, 6  # Spans over midnight

# Segregate data into different time periods
df1_morning = df1[(df1['ArrivalHour'] >= morning_start) & (df1['ArrivalHour'] < morning_end)]
df1_afternoon = df1[(df1['ArrivalHour'] >= afternoon_start) & (df1['ArrivalHour'] < afternoon_end)]
df1_early_evening = df1[(df1['ArrivalHour'] >= early_evening_start) & (df1['ArrivalHour'] < early_evening_end)]
df1_late_evening_night = df1[(df1['ArrivalHour'] >= late_evening_start) | (df1['ArrivalHour'] < late_evening_end)]


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
def distribution_params_at(df1): #Gamma distribution is best fit
    df1 = df1.dropna(subset=['UTCTransactionStart'])      # Handle NaN values
    df1_sorted = df1.sort_values(by='UTCTransactionStart')      # Sort by transaction start time for chronical order
    df1_sorted['InterarrivalTime'] = df1_sorted['UTCTransactionStart'].diff().dt.total_seconds().div(60)  #Calculate the interarrival times in minutes
    df1_sorted['InterarrivalTime'] = df1_sorted['InterarrivalTime'].replace(0, np.nan)      # Handle zero values, cacn be immediate consecutive transactions 
    df1_sorted['InterarrivalTime'] = df1_sorted['InterarrivalTime'].ffill()  # Forward fill to maintain continuity for immediate consecutive transactions
    df1_sorted = df1_sorted[np.isfinite(df1_sorted['InterarrivalTime'])]        # Check and remove infinite values needed for statistical modelling
    df1_sorted['InterarrivalTime'] = df1_sorted['InterarrivalTime'].clip(upper=1440)        # Clip the range
    df1_sorted['Date'] = df1_sorted['UTCTransactionStart'].dt.date      # Extract the date
    daily_interarrival = df1_sorted.groupby('Date')['InterarrivalTime'].apply(list)     # Aggregate interarrival times by day
    average_day_interarrival = np.concatenate(daily_interarrival.values)        # Combine for an average day
    time_diffs_at = np.log(average_day_interarrival + 1)  # Add 1 to avoid log(0)  Log transformation for skewness

    try:
        params_gamma_at = stats.gamma.fit(time_diffs_at, floc=0)
        params_expon_at = stats.expon.fit(time_diffs_at, floc=0)
        params_lognorm_at = stats.lognorm.fit(time_diffs_at, floc=0)
    except ValueError as e:
        print(f"Error fitting distributions: {e}")
        return None

    return params_gamma_at, params_expon_at, params_lognorm_at, time_diffs_at



def available_service_time_distribution(df1): #Lognorm distribution is best fit, only Gamma distribution is best fit before filtering.
    df1 = df1.dropna(subset=['UTCTransactionStart', 'UTCTransactionStop'])      # Drop NaN values in relevant columns
    df1_sorted = df1.sort_values(by='UTCTransactionStart')      # Sort by transaction start time
    df1_sorted['AvailableServiceTime'] = (df1_sorted['UTCTransactionStop'] - df1_sorted['UTCTransactionStart']).dt.total_seconds() / 60     # Calculate the available service time in minutes
    df1_sorted['AvailableServiceTime'] = df1_sorted['AvailableServiceTime'].clip(upper=1440)        # Handle outliers (e.g., extremely long service times)   Assuming service times longer than a day (1440 minutes) are outliers
    df1_sorted = df1_sorted[np.isfinite(df1_sorted['AvailableServiceTime'])]        # Check and remove infinite values
    avg_service_time_per_minute = df1_sorted.groupby(df1_sorted['UTCTransactionStart'].dt.minute)['AvailableServiceTime'].mean()    # Group by ArrivalMinute and calculate the average available service time per minute
    
    try:
        params_gamma_ast = stats.gamma.fit(avg_service_time_per_minute, floc=0)
        params_expon_ast = stats.expon.fit(avg_service_time_per_minute, floc=0)
        params_lognorm_ast = stats.lognorm.fit(avg_service_time_per_minute, floc=0)
    except ValueError as e:
        print(f"Error fitting distributions: {e}")
        return None

    return params_gamma_ast, params_expon_ast, params_lognorm_ast, avg_service_time_per_minute


def distribution_params_te(df1): #lognormal distribution is best fit
    df1 = df1.dropna(subset=['TotalEnergy', 'UTCTransactionStart']) # Drop NaN values in TotalEnergy column
    df1_sorted = df1.sort_values(by='UTCTransactionStart')      # Sort by transaction start time
    df1_sorted['TotalEnergy'] = df1_sorted['TotalEnergy'].clip(lower=1, upper=70)   # Clip the TotalEnergy values to be between 1 and 70 kW
    df1_sorted['ArrivalMinute'] = df1_sorted['UTCTransactionStart'].dt.minute       # Group by ArrivalMinute and calculate the average total energy per minute
    avg_total_energy_per_minute = df1_sorted.groupby('ArrivalMinute')['TotalEnergy'].mean()

    try:
        params_gamma_te = stats.gamma.fit(avg_total_energy_per_minute, floc=0)
        params_expon_te = stats.expon.fit(avg_total_energy_per_minute, floc=0)
        params_lognorm_te = stats.lognorm.fit(avg_total_energy_per_minute, floc=0)
    except ValueError as e:
        print(f"Error fitting distributions: {e}")
        return None

    return params_gamma_te, params_expon_te, params_lognorm_te, avg_total_energy_per_minute

def statistical_metrics(data, params_gamma, params_expon, params_lognorm):
    data_filtered = data[data > 0]  # Ensure data is positive for these distributions
    ks_stat_gamma, p_value_gamma = stats.kstest(data_filtered, 'gamma', params_gamma)
    ks_stat_expon, p_value_expon = stats.kstest(data_filtered, 'expon', params_expon)
    ks_stat_lognorm, p_value_lognorm = stats.kstest(data_filtered, 'lognorm', params_lognorm)

    aic_gamma = calculate_aic_bic(data_filtered, 'gamma', params_gamma)
    aic_expon = calculate_aic_bic(data_filtered, 'expon', params_expon)
    aic_lognorm = calculate_aic_bic(data_filtered, 'lognorm', params_lognorm)

    bic_gamma = calculate_aic_bic(data_filtered, 'gamma', params_gamma, use_bic=True)
    bic_expon = calculate_aic_bic(data_filtered, 'expon', params_expon, use_bic=True)
    bic_lognorm = calculate_aic_bic(data_filtered, 'lognorm', params_lognorm, use_bic=True)

    return {
        'gamma': {'ks_stat': ks_stat_gamma, 'p_value': p_value_gamma, 'aic': aic_gamma, 'bic': bic_gamma},
        'expon': {'ks_stat': ks_stat_expon, 'p_value': p_value_expon, 'aic': aic_expon, 'bic': bic_expon},
        'lognorm': {'ks_stat': ks_stat_lognorm, 'p_value': p_value_lognorm, 'aic': aic_lognorm, 'bic': bic_lognorm}
    }


def calculate_aic_bic(data, dist_name, params, use_bic=False):
    k = len(params)
    llk = np.sum(getattr(stats, dist_name).logpdf(data, *params))
    return -2 * llk + k * (np.log(len(data)) if use_bic else 2)


#BEST FIT IS GAMMA FOR ARRIVAL TIME FULL Day
params_gamma_at, params_expon_at, params_lognorm_at, time_diffs_at = distribution_params_at(df1)
metrics_at = statistical_metrics(time_diffs_at, params_gamma_at, params_expon_at, params_lognorm_at)

# Calculate interArrival Time (AT) distribution parameters and metrics per time segment
params_gamma_at_morning, params_expon_at_morning, params_lognorm_at_morning, time_diffs_at_morning = distribution_params_at(df1_morning)
metrics_at_morning = statistical_metrics(time_diffs_at_morning, params_gamma_at_morning, params_expon_at_morning, params_lognorm_at_morning)

params_gamma_at_afternoon, params_expon_at_afternoon, params_lognorm_at_afternoon, time_diffs_at_afternoon = distribution_params_at(df1_afternoon)
metrics_at_afternoon = statistical_metrics(time_diffs_at_afternoon, params_gamma_at_afternoon, params_expon_at_afternoon, params_lognorm_at_afternoon)

params_gamma_at_early_evening, params_expon_at_early_evening, params_lognorm_at_early_evening, time_diffs_at_early_evening = distribution_params_at(df1_early_evening)
metrics_at_early_evening = statistical_metrics(time_diffs_at_early_evening, params_gamma_at_early_evening, params_expon_at_early_evening, params_lognorm_at_early_evening)

params_gamma_at_late_evening_night, params_expon_at_late_evening_night, params_lognorm_at_late_evening_night, time_diffs_at_late_evening_night = distribution_params_at(df1_late_evening_night)
metrics_at_late_evening_night = statistical_metrics(time_diffs_at_late_evening_night, params_gamma_at_late_evening_night, params_expon_at_late_evening_night, params_lognorm_at_late_evening_night)

# Calculate Available Service Time (AST) distribution parameters and metrics per time segment
params_gamma_ast_morning, params_expon_ast_morning, params_lognorm_ast_morning, avg_service_time_per_minute_morning = available_service_time_distribution(df1_morning)
metrics_ast_morning = statistical_metrics(avg_service_time_per_minute_morning, params_gamma_ast_morning, params_expon_ast_morning, params_lognorm_ast_morning)

params_gamma_ast_afternoon, params_expon_ast_afternoon, params_lognorm_ast_afternoon, avg_service_time_per_afternoon = available_service_time_distribution(df1_afternoon)
metrics_ast_afternoon = statistical_metrics(avg_service_time_per_afternoon, params_gamma_ast_afternoon, params_expon_ast_afternoon, params_lognorm_at_afternoon)

params_gamma_ast_early_evening, params_expon_ast_early_evening, params_lognorm_ast_early_evening, avg_service_time_per_minute_early_evening = available_service_time_distribution(df1_early_evening)
metrics_ast_early_evening = statistical_metrics(avg_service_time_per_minute_early_evening, params_gamma_ast_early_evening, params_expon_ast_early_evening, params_lognorm_ast_early_evening)

params_gamma_ast_late_evening_night, params_expon_ast_late_evening_night, params_lognorm_ast_late_evening_night, avg_service_time_per_minute_late_evening_night = available_service_time_distribution(df1_late_evening_night)
metrics_ast_late_evening_night = statistical_metrics(avg_service_time_per_minute_late_evening_night, params_gamma_at_late_evening_night, params_expon_at_late_evening_night, params_lognorm_at_late_evening_night)

# Calculate Total Energy (TE) Demand distribution parameters and metrics per time segment
params_gamma_te_morning, params_expon_te_morning, params_lognorm_te_morning, avg_total_energy_per_minute_morning = distribution_params_te(df1_morning)
metrics_te_morning = statistical_metrics(avg_total_energy_per_minute_morning, params_gamma_te_morning, params_expon_te_morning, params_lognorm_te_morning)

params_gamma_te_afternoon, params_expon_te_afternoon, params_lognorm_te_afternoon, avg_total_energy_per_minute_afternoon = distribution_params_te(df1_afternoon)
metrics_te_afternoon = statistical_metrics(avg_total_energy_per_minute_afternoon, params_gamma_te_afternoon, params_expon_te_afternoon, params_lognorm_te_afternoon)

params_gamma_te_early_evening, params_expon_te_early_evening, params_lognorm_te_early_evening, avg_total_energy_per_minute_early_evening = distribution_params_te(df1_early_evening)
metrics_te_early_evening = statistical_metrics(avg_total_energy_per_minute_early_evening, params_gamma_te_early_evening, params_expon_te_early_evening, params_lognorm_te_early_evening)

params_gamma_te_late_evening_night, params_expon_te_late_evening_night, params_lognorm_te_late_evening_night, avg_total_energy_per_minute_late_evening_night = distribution_params_te(df1_late_evening_night)
metrics_te_late_evening_night = statistical_metrics(avg_total_energy_per_minute_late_evening_night, params_gamma_te_late_evening_night, params_expon_te_late_evening_night, params_lognorm_te_late_evening_night)

# Compiling results into a dictionary
results = {
    'Morning': { #For Mornings: ALL Gamma distributions
        'AT_params_gamma': params_gamma_at_morning,
        'AT_params_expon': params_expon_at_morning,
        'AT_params_lognorm': params_lognorm_at_morning,
        'AT_metrics': metrics_at_morning,
        'AST_params_gamma': params_gamma_ast_morning,
        'AST_params_expon': params_expon_ast_morning,
        'AST_params_lognorm': params_lognorm_ast_morning,
        'AST_metrics': metrics_ast_morning,
        'TE_params_gamma': params_gamma_te_morning,
        'TE_params_expon': params_expon_te_morning,
        'TE_params_lognorm': params_lognorm_te_morning,
        'TE_metrics': metrics_te_morning
    },
    'Afternoon': { #For Afternoons: Lognorm fits AT the best. None fit the AST best, but Gamma is the closest. Lognormal best for TE.
        'AT_params_gamma': params_gamma_at_afternoon,
        'AT_params_expon': params_expon_at_afternoon,
        'AT_params_lognorm': params_lognorm_at_afternoon,
        'AT_metrics': metrics_at_afternoon,
        'AST_params_gamma': params_gamma_ast_afternoon,
        'AST_params_expon': params_expon_ast_afternoon,
        'AST_params_lognorm': params_lognorm_ast_afternoon,
        'AST_metrics': metrics_ast_afternoon,
        'TE_params_gamma': params_gamma_te_afternoon,
        'TE_params_expon': params_expon_te_afternoon,
        'TE_params_lognorm': params_lognorm_te_afternoon,
        'TE_metrics': metrics_te_afternoon
    },
    'Early Evening': { #For Early Evening: Gamma has the best fit for AT and AST. Lognorm best for TE.
        'AT_params_gamma': params_gamma_at_early_evening,
        'AT_params_expon': params_expon_at_early_evening,
        'AT_params_lognorm': params_lognorm_at_early_evening,
        'AT_metrics': metrics_at_early_evening,
        'AST_params_gamma': params_gamma_ast_early_evening,
        'AST_params_expon': params_expon_ast_early_evening,
        'AST_params_lognorm': params_lognorm_ast_early_evening,
        'AST_metrics': metrics_ast_early_evening,
        'TE_params_gamma': params_gamma_te_early_evening,
        'TE_params_expon': params_expon_te_early_evening,
        'TE_params_lognorm': params_lognorm_te_early_evening,
        'TE_metrics': metrics_te_early_evening
    },
    'Late Evening and Night': { #For Late Evening and Night: Lognormal is the best fit for AT. All are poor for AST, but Lognorm least poor.Gamma best fit for TE.
        'AT_params_gamma': params_gamma_at_late_evening_night,
        'AT_params_expon': params_expon_at_late_evening_night,
        'AT_params_lognorm': params_lognorm_at_late_evening_night,
        'AT_metrics': metrics_at_late_evening_night,
        'AST_params_gamma': params_gamma_ast_late_evening_night,
        'AST_params_expon': params_expon_ast_late_evening_night,
        'AST_params_lognorm': params_lognorm_ast_late_evening_night,
        'AST_metrics': metrics_ast_late_evening_night,
        'TE_params_gamma': params_gamma_te_late_evening_night,
        'TE_params_expon': params_expon_te_late_evening_night,
        'TE_params_lognorm': params_lognorm_te_late_evening_night,
        'TE_metrics': metrics_te_late_evening_night
    }
}

# Print the results
for time_segment, segment_data in results.items():
    print(f"\n{time_segment} Segment:")
    for distribution_type in ['AT', 'AST', 'TE']:
        print(f"  {distribution_type}:")
        params_gamma = segment_data.get(f'{distribution_type}_params_gamma', 'N/A')
        params_expon = segment_data.get(f'{distribution_type}_params_expon', 'N/A')
        params_lognorm = segment_data.get(f'{distribution_type}_params_lognorm', 'N/A')
        metrics = segment_data.get(f'{distribution_type}_metrics', {})
        print(f"    Params - Gamma: {params_gamma}")
        print(f"    Params - Expon: {params_expon}")
        print(f"    Params - Lognorm: {params_lognorm}")
        print(f"    Metrics: ")
        for dist, metrics_data in metrics.items():
            print(f"      {dist.capitalize()}: KS Stat: {metrics_data['ks_stat']}, P-value: {metrics_data['p_value']}, AIC: {metrics_data['aic']}, BIC: {metrics_data['bic']}")


#INCONCLUSIVE: after filtering <=70 better fit lognorm, otherwise gamma
params_gamma_ast, params_expon_ast, params_lognorm_ast, avg_service_time_per_minute = available_service_time_distribution(df1)
metrics_ast = statistical_metrics(avg_service_time_per_minute, params_gamma_ast, params_expon_ast, params_lognorm_ast)

#BEST FIT IS LOGNORMAL FOR TOTAL ENERGY
params_gamma_te, params_expon_te, params_lognorm_te, avg_total_energy_per_minute = distribution_params_te(df1)
metrics_te = statistical_metrics(avg_total_energy_per_minute, params_gamma_te, params_expon_te, params_lognorm_te)

mean_arrival_time = gamma.mean(*params_gamma_at)

# Correcting the statistical metrics calculation

print("Gamma distribution parameters Arrival time:", params_gamma_at)
print("Lognorm distribution parameters Total Energt:", params_lognorm_te)
print("Lognorm distribution parameters Available Service Time:", params_lognorm_ast)
print("Metric Total Energy distributions", metrics_te)
print("Metric Available Service Time distributions", metrics_ast)
print("Metric Arrival Time distributions", metrics_at)
print("Mean Arrival Time (minutes):", mean_arrival_time)


#save params_gamma LOOK! I AM A PICKLE
with open('params_gamma_at.pkl', 'wb') as f:
    pickle.dump(params_gamma_at, f)

with open('params_gamma_ast.pkl', 'wb') as f:
    pickle.dump(params_gamma_ast, f)

with open('params_lognorm_ast.pkl', 'wb') as f:
    pickle.dump(params_lognorm_ast, f)

with open('params_lognorm_te.pkl', 'wb') as f:
    pickle.dump(params_lognorm_te, f)

#Time segments pickles
with open('params_gamma_at_morning.pkl', 'wb') as f:
    pickle.dump(params_gamma_at_morning, f)

with open('params_lognorm_at_morning.pkl', 'wb') as f:
    pickle.dump(params_lognorm_at_morning, f)

with open('params_gamma_ast_morning.pkl', 'wb') as f:
    pickle.dump(params_gamma_ast_morning, f)

with open('params_gamma_te_morning.pkl', 'wb') as f:
    pickle.dump(params_gamma_te_morning, f)

with open('params_lognorm_te_morning.pkl', 'wb') as f:
    pickle.dump(params_lognorm_te_morning, f)

#Plots
    # Generate points on the x-axis for plotting the PDFs
x_gamma_at_morning = np.linspace(start=0, stop=max(time_diffs_at_morning), num=1400)
x_expon_at_morning = np.linspace(start=0, stop=max(time_diffs_at_morning), num=1400)
x_lognorm_at_morning = np.linspace(start=0, stop=max(time_diffs_at_morning), num=1400)

# Calculate the PDFs for the fitted distributions
pdf_gamma_at_morning = stats.gamma.pdf(x_gamma_at_morning, *params_gamma_at_morning)
pdf_expon_at_morning = stats.expon.pdf(x_expon_at_morning, *params_expon_at_morning)
pdf_lognorm_at_morning = stats.lognorm.pdf(x_lognorm_at_morning, *params_lognorm_at_morning)

# Plot the histogram of the empirical data
plt.figure(figsize=(15, 6))
sns.histplot(time_diffs_at_morning, bins=30, kde=False, stat='density', label='Empirical Data Morning segment', color='#FBD5EC')

# Plot the PDFs
plt.plot(x_gamma_at_morning, pdf_gamma_at_morning, label='Fitted Gamma Distribution', color='#B9D531')
plt.plot(x_expon_at_morning, pdf_expon_at_morning, label='Fitted Exponential Distribution', color='#EC008C')
plt.plot(x_lognorm_at_morning, pdf_lognorm_at_morning, label='Fitted Lognormal Distribution', color='#696A6C')

# Finalize the plot
plt.xlabel('Time Difference Between Arrivals (Log Transformed Minutes)')
plt.ylabel('Density')
plt.title('Distribution of EV Charging Station Arrivals in the Morning and Model Fits')
plt.legend()
plt.show()
