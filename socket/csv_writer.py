#!/usr/bin/env python3
"""
CSV Writer for Option Divergence Analysis
Handles data export and formatting for analysis results
"""

import csv
import pandas as pd
from datetime import datetime
import os

class DivergenceCSVWriter:
    """CSV writer for divergence analysis results"""
    
    def __init__(self, output_dir='../data'):
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def write_divergence_data(self, divergence_results, filename=None):
        """
        Write divergence analysis results to CSV
        
        Args:
            divergence_results: List of dictionaries containing divergence metrics
            filename: Optional filename, auto-generated if None
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'divergence_analysis_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not divergence_results:
            print("No divergence results to write")
            return None
        
        #------------Extract fieldnames from first result
        fieldnames = list(divergence_results[0].keys())
        
        #------------Write CSV file
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in divergence_results:
                #----Handle nested dictionaries by flattening
                flattened_result = self._flatten_dict(result)
                writer.writerow(flattened_result)
        
        print(f"Divergence data written to {filepath}")
        return filepath
    
    def write_signal_performance(self, signals_data, filename=None):
        """
        Write signal performance metrics to CSV
        
        Args:
            signals_data: Dictionary containing signal performance metrics
            filename: Optional filename, auto-generated if None
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'signal_performance_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        #------------Convert to DataFrame for easier handling
        df = pd.DataFrame([signals_data])
        df.to_csv(filepath, index=False)
        
        print(f"Signal performance data written to {filepath}")
        return filepath
    
    def write_statistical_results(self, stats_results, filename=None):
        """
        Write statistical test results to CSV
        
        Args:
            stats_results: Dictionary containing statistical test results
            filename: Optional filename, auto-generated if None
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'statistical_results_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        #------------ Flatten nested results
        flattened_results = []
        for test_name, results in stats_results.items():
            if isinstance(results, dict):
                flattened_result = {'test_name': test_name}
                flattened_result.update(self._flatten_dict(results))
                flattened_results.append(flattened_result)
            else:
                flattened_results.append({'test_name': test_name, 'result': results})
        
        #------------ Write to CSV
        if flattened_results:
            df = pd.DataFrame(flattened_results)
            df.to_csv(filepath, index=False)
            print(f"Statistical results written to {filepath}")
            return filepath
        
        return None
    
    def write_extreme_events(self, extreme_events, filename=None):
        """
        Write extreme divergence events to CSV
        
        Args:
            extreme_events: List of extreme event dictionaries
            filename: Optional filename, auto-generated if None
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'extreme_events_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not extreme_events:
            print("No extreme events to write")
            return None
        
        #------------ Convert to DataFrame and write
        df = pd.DataFrame(extreme_events)
        df.to_csv(filepath, index=False)
        
        print(f"Extreme events written to {filepath}")
        return filepath
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten nested dictionary for CSV writing
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                #--------Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def append_realtime_data(self, data_point, filename='realtime_divergence.csv'):
        """
        Append real-time data point to existing CSV file
        
        Args:
            data_point: Dictionary containing single data point
            filename: CSV filename to append to
        """
        filepath = os.path.join(self.output_dir, filename)
        
        #------------Check if file exists
        file_exists = os.path.isfile(filepath)
        
        #------------Flatten the data point
        flattened_data = self._flatten_dict(data_point)
        
        #------------Write or append to CSV
        with open(filepath, 'a', newline='') as csvfile:
            fieldnames = list(flattened_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            #------------Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(flattened_data)
    
    def create_summary_report(self, analysis_results, filename=None):
        """
        Create summary report CSV with key metrics
        
        Args:
            analysis_results: Dictionary containing all analysis results
            filename: Optional filename, auto-generated if None
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'analysis_summary_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        #------------Extract key summary metrics
        summary_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_observations': analysis_results.get('total_observations', 0),
            'mean_price_divergence': analysis_results.get('mean_price_divergence', 0),
            'mean_iv_divergence': analysis_results.get('mean_iv_divergence', 0),
            'total_signals': analysis_results.get('total_signals', 0),
            'signal_accuracy': analysis_results.get('signal_accuracy', 0),
            'extreme_events_count': analysis_results.get('extreme_events_count', 0),
            'cointegration_p_value': analysis_results.get('cointegration_p_value', 1.0),
            'statistical_significance': analysis_results.get('statistical_significance', False)
        }
        
        #------------Write summary to CSV
        df = pd.DataFrame([summary_data])
        df.to_csv(filepath, index=False)
        
        print(f"Analysis summary written to {filepath}")
        return filepath

#----------------Example usage
if __name__ == "__main__":
    writer = DivergenceCSVWriter()
    
    #------------Example divergence data
    sample_data = [
        {
            'timestamp': datetime.now(),
            'price_divergence': 0.123,
            'iv_divergence': -0.045,
            'parity_divergence': 0.067,
            'signal': 1,
            'confidence': 0.85
        }
    ]
    
    writer.write_divergence_data(sample_data)