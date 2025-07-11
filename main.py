import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.title("24-Month Forecast Simulation")

# File upload
st.subheader("Data Upload")
st.write("Upload CSV file with historical data including shelf life and channel inventory")
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read and validate data
        df = pd.read_csv(uploaded_file)
        
        # Display raw data
        st.subheader("Raw Data Preview")
        st.write(df.head())
        
        # Data validation
        required_columns = ['date', 'value', 'shelf_life_days', 'channel_inventory']
        if not all(col in df.columns for col in required_columns):
            st.error("Error: CSV must contain 'date', 'value', 'shelf_life_days', and 'channel_inventory' columns")
        else:
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Calculate inventory age and remaining shelf life
            df['inventory_age'] = (df['date'].max() - df['date']).dt.days
            df['remaining_shelf_life'] = df['shelf_life_days'] - df['inventory_age']
            
            # Calculate inventory turnover rate
            df['turnover_rate'] = df['value'] / df['channel_inventory']
            
            # Basic statistics
            st.subheader("Data Statistics")
            stats = {
                "Date Range": f"{df['date'].min()} to {df['date'].max()}",
                "Average Demand": f"{df['value'].mean():.2f}",
                "Average Channel Inventory": f"{df['channel_inventory'].mean():.2f}",
                "Average Turnover Rate": f"{df['turnover_rate'].mean():.2f}",
                "Average Shelf Life": f"{df['shelf_life_days'].mean():.0f} days"
            }
            st.write(stats)
            
            # Forecast Parameters
            st.subheader("Forecast Parameters")
            
            # Market Parameters
            st.write("Market Parameters")
            col1, col2 = st.columns(2)
            with col1:
                growth_rate = st.slider("Annual Growth Rate (%)", -20, 20, 5)
                seasonality = st.slider("Seasonality Impact (%)", 0, 50, 10)
            with col2:
                volatility = st.slider("Volatility (%)", 0, 30, 5)
                confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
            
            # Inventory Parameters
            st.write("Inventory Parameters")
            col3, col4 = st.columns(2)
            with col3:
                shelf_life_impact = st.slider("Shelf Life Impact on Demand (%)", 0, 100, 30,
                                           help="How much remaining shelf life affects demand")
                safety_stock = st.slider("Safety Stock (% of demand)", 0, 100, 20)
            with col4:
                inventory_impact = st.slider("Channel Inventory Impact (%)", 0, 100, 40,
                                          help="How much channel inventory affects demand")
                reorder_point = st.slider("Reorder Point (% of safety stock)", 0, 200, 100)
            
            # Generate forecast
            last_date = df['date'].max()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=24, freq='ME')
            
            # Calculate base trend from historical data
            historical_growth = df['value'].pct_change().mean()
            last_value = df['value'].iloc[-1]
            
            # Calculate inventory factors
            avg_turnover = df['turnover_rate'].mean()
            current_turnover = df['turnover_rate'].iloc[-1]
            inventory_factor = 1 + (avg_turnover - current_turnover) * (inventory_impact/100)
            
            # Calculate shelf life factor
            remaining_life_ratio = df['remaining_shelf_life'].iloc[-1] / df['shelf_life_days'].iloc[-1]
            shelf_life_factor = 1 - ((1 - remaining_life_ratio) * (shelf_life_impact/100))
            
            # Calculate base forecast incorporating historical patterns
            monthly_growth = (1 + growth_rate/100) ** (1/12) - 1
            trend = np.array([last_value * (1 + monthly_growth) ** i for i in range(24)])
            
            # Add seasonality
            seasonal_factor = seasonality/100
            seasonal_pattern = np.sin(np.arange(24) * 2 * np.pi / 12)
            seasonality_component = trend * seasonal_factor * seasonal_pattern
            
            # Add volatility
            volatility_factor = volatility/100
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, volatility_factor, 24) * trend
            
            # Combine components for initial demand forecast
            base_forecast = trend + seasonality_component + noise
            
            # Apply inventory and shelf life factors
            demand_forecast = base_forecast * inventory_factor * shelf_life_factor
            
            # Calculate inventory metrics
            safety_stock_level = safety_stock/100 * demand_forecast
            reorder_threshold = reorder_point/100 * safety_stock_level
            
            # Initialize tracking arrays
            inventory_levels = np.zeros(24)
            expired_inventory = np.zeros(24)
            new_orders = np.zeros(24)
            shelf_life_remaining = np.zeros(24)
            turnover_rates = np.zeros(24)
            
            # Set initial values
            inventory_levels[0] = df['channel_inventory'].iloc[-1]
            shelf_life_remaining[0] = df['remaining_shelf_life'].iloc[-1]
            
            # Calculate future inventory and demand
            for i in range(1, 24):
                # Update shelf life
                shelf_life_remaining[i] = max(0, shelf_life_remaining[i-1] - 30)
                
                # Calculate turnover rate
                turnover_rates[i-1] = demand_forecast[i-1] / inventory_levels[i-1] if inventory_levels[i-1] > 0 else 0
                
                # Adjust demand based on current conditions
                shelf_life_factor = 1 - ((1 - shelf_life_remaining[i]/df['shelf_life_days'].mean()) * (shelf_life_impact/100))
                inventory_factor = 1 + (avg_turnover - turnover_rates[i-1]) * (inventory_impact/100)
                demand_forecast[i] = demand_forecast[i] * shelf_life_factor * inventory_factor
                
                # Handle expired inventory
                if shelf_life_remaining[i] <= 0:
                    expired_inventory[i] = inventory_levels[i-1]
                    inventory_levels[i-1] = 0
                
                # Calculate remaining inventory
                remaining_inventory = max(0, inventory_levels[i-1] - demand_forecast[i-1] - expired_inventory[i])
                
                # Place new order if needed
                if remaining_inventory < reorder_threshold[i]:
                    new_orders[i] = safety_stock_level[i] - remaining_inventory
                    inventory_levels[i] = remaining_inventory + new_orders[i]
                    shelf_life_remaining[i] = df['shelf_life_days'].mean()  # Reset shelf life for new inventory
                else:
                    inventory_levels[i] = remaining_inventory
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'demand': demand_forecast,
                'inventory': inventory_levels,
                'expired': expired_inventory,
                'new_orders': new_orders,
                'safety_stock': safety_stock_level,
                'reorder_point': reorder_threshold,
                'shelf_life_remaining': shelf_life_remaining,
                'turnover_rate': turnover_rates
            })
            
            # Calculate confidence intervals
            z_score = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
            std_dev = volatility_factor * trend
            forecast_df['lower_bound'] = demand_forecast - z_score * std_dev
            forecast_df['upper_bound'] = demand_forecast + z_score * std_dev
            
            # Prepare data for visualization
            historical_df = df.copy()
            historical_df['type'] = 'Historical'
            forecast_view_df = forecast_df[['date', 'demand']].rename(columns={'demand': 'value'})
            forecast_view_df['type'] = 'Forecast'
            combined_df = pd.concat([historical_df, forecast_view_df])
            
            # Visualizations
            st.subheader("Forecast Results")
            
            # Demand Forecast Chart
            st.write("Demand Forecast with Inventory and Shelf Life Impact")
            base = alt.Chart(combined_df).encode(x='date:T')
            
            historical_line = base.mark_line(color='blue').encode(
                y='value:Q',
                tooltip=['date:T', 'value:Q', 'type:N']
            ).transform_filter(alt.datum.type == 'Historical')
            
            forecast_line = base.mark_line(color='red', strokeDash=[5,5]).encode(
                y='value:Q',
                tooltip=['date:T', 'value:Q', 'type:N']
            ).transform_filter(alt.datum.type == 'Forecast')
            
            confidence_area = alt.Chart(forecast_df).mark_area(opacity=0.2).encode(
                x='date:T',
                y='lower_bound:Q',
                y2='upper_bound:Q',
                tooltip=['date:T', 'lower_bound:Q', 'upper_bound:Q']
            )
            
            demand_chart = (historical_line + forecast_line + confidence_area).properties(
                width=800,
                height=400,
                title='Demand Forecast Incorporating Inventory and Shelf Life Factors'
            )
            
            st.altair_chart(demand_chart)
            
            # Inventory and Shelf Life Chart
            st.write("Inventory Management and Shelf Life")
            inventory_base = alt.Chart(forecast_df).encode(x='date:T')
            
            inventory_line = inventory_base.mark_line(color='green').encode(
                y='inventory:Q',
                tooltip=['date:T', 'inventory:Q', 'turnover_rate:Q']
            )
            
            shelf_life_line = inventory_base.mark_line(color='purple', strokeDash=[5,5]).encode(
                y='shelf_life_remaining:Q',
                tooltip=['date:T', 'shelf_life_remaining:Q']
            )
            
            expired_bars = inventory_base.mark_bar(color='red', opacity=0.5).encode(
                y='expired:Q',
                tooltip=['date:T', 'expired:Q']
            )
            
            inventory_chart = (inventory_line + shelf_life_line + expired_bars).properties(
                width=800,
                height=400,
                title='Inventory Levels, Shelf Life, and Expired Products'
            )
            
            st.altair_chart(inventory_chart)
            
            # Display forecast statistics
            st.subheader("Forecast Statistics")
            forecast_stats = {
                "Demand Impact Factors": {
                    "Inventory Impact": f"{inventory_factor:.2f}x",
                    "Shelf Life Impact": f"{shelf_life_factor:.2f}x",
                    "Combined Impact": f"{(inventory_factor * shelf_life_factor):.2f}x"
                },
                "Forecast Metrics": {
                    "Average Forecasted Demand": f"{demand_forecast.mean():.2f}",
                    "Average Inventory Level": f"{inventory_levels.mean():.2f}",
                    "Total Expired Products": f"{expired_inventory.sum():.2f}",
                    "Average Turnover Rate": f"{turnover_rates.mean():.2f}",
                    "Confidence Level": f"{confidence_level}%"
                }
            }
            st.write(forecast_stats)
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file with required columns: 'date', 'value', 'shelf_life_days', and 'channel_inventory'")
    
    # Sample data format
    st.subheader("Sample Data Format")
    sample_data = pd.DataFrame({
        'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
        'value': [1000, 1100, 1050],
        'shelf_life_days': [180, 180, 180],
        'channel_inventory': [2000, 1800, 1900]
    })
    st.write(sample_data)