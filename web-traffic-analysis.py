import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def analyze_pageviews(df):
    # Total pageviews
    total_pageviews = df[df['event'] == 'pageview']['count'].sum()
    
    # Daily pageviews
    daily_pageviews = df[df['event'] == 'pageview'].groupby('date')['count'].sum()
    avg_daily_pageviews = daily_pageviews.mean()
    
    print("\n1. Pageview Analysis:")
    print(f"Total pageviews: {total_pageviews:,}")
    print(f"Average daily pageviews: {avg_daily_pageviews:,.2f}")
    
    # Visualize daily pageviews
    plt.figure(figsize=(12, 6))
    daily_pageviews.plot(kind='bar')
    plt.title('Daily Pageviews')
    plt.xlabel('Date')
    plt.ylabel('Number of Pageviews')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return total_pageviews, avg_daily_pageviews

def analyze_events(df):
    # Event distribution
    event_counts = df.groupby('event')['count'].sum().sort_values(ascending=False)
    
    print("\n2. Event Distribution:")
    print(event_counts)
    
    # Visualize event distribution
    plt.figure(figsize=(10, 6))
    event_counts.plot(kind='bar')
    plt.title('Distribution of Events')
    plt.xlabel('Event Type')
    plt.ylabel('Total Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return event_counts

def analyze_geography(df):
    # Geographic distribution of pageviews
    geo_pageviews = df[df['event'] == 'pageview'].groupby('geo')['count'].sum().sort_values(ascending=False)
    
    print("\n3. Geographic Distribution:")
    print(geo_pageviews)
    
    # Visualize geographic distribution
    plt.figure(figsize=(12, 6))
    geo_pageviews.plot(kind='bar')
    plt.title('Geographic Distribution of Pageviews')
    plt.xlabel('Country')
    plt.ylabel('Number of Pageviews')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return geo_pageviews

def analyze_ctr(df):
    # Calculate CTR per link
    link_metrics = df.pivot_table(
        index='link',
        columns='event',
        values='count',
        aggfunc='sum'
    ).fillna(0)
    
    link_metrics['ctr'] = (link_metrics['click'] / link_metrics['pageview'] * 100)
    overall_ctr = (link_metrics['click'].sum() / link_metrics['pageview'].sum() * 100)
    
    print("\n4. Click-Through Rate Analysis:")
    print(f"Overall CTR: {overall_ctr:.2f}%")
    print("\nCTR by Link:")
    print(link_metrics['ctr'].sort_values(ascending=False))
    
    # Visualize CTR distribution
    plt.figure(figsize=(12, 6))
    link_metrics['ctr'].sort_values().plot(kind='bar')
    plt.title('CTR Distribution by Link')
    plt.xlabel('Link')
    plt.ylabel('CTR (%)')
    plt.axhline(y=overall_ctr, color='r', linestyle='--', label='Overall CTR')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return link_metrics, overall_ctr

def analyze_correlation(df):
    # Prepare data for correlation analysis
    link_metrics = df.pivot_table(
        index='link',
        columns='event',
        values='count',
        aggfunc='sum'
    ).fillna(0)
    
    # Pearson correlation
    correlation, p_value = stats.pearsonr(link_metrics['click'], link_metrics['preview'])
    
    # Spearman rank correlation
    spearman_corr, spearman_p = stats.spearmanr(link_metrics['click'], link_metrics['preview'])
    
    print("\n5. Correlation Analysis:")
    print(f"Pearson correlation coefficient: {correlation:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Spearman rank correlation: {spearman_corr:.3f}")
    print(f"Spearman p-value: {spearman_p:.3f}")
    
    # Visualize correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(link_metrics['preview'], link_metrics['click'])
    plt.xlabel('Previews')
    plt.ylabel('Clicks')
    plt.title('Correlation between Clicks and Previews')
    
    # Add regression line
    z = np.polyfit(link_metrics['preview'], link_metrics['click'], 1)
    p = np.poly1d(z)
    plt.plot(link_metrics['preview'], p(link_metrics['preview']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    return correlation, p_value, spearman_corr, spearman_p

def main():
    # Load data
    df = load_data('traffic.csv')
    
    # Run all analyses
    pageview_results = analyze_pageviews(df)
    event_results = analyze_events(df)
    geo_results = analyze_geography(df)
    ctr_results = analyze_ctr(df)
    correlation_results = analyze_correlation(df)

if __name__ == "__main__":
    main()
