# Energy Consumption Analysis Dashboard

A comprehensive Streamlit-based dashboard for analyzing energy consumption patterns using machine learning and statistical methods, with advanced anomaly detection capabilities.

![Dashboard Overview](image.png)

_Main dashboard interface showing the multi-tab analysis system_

## 🚀 Features

### Core Functionality

- **Multi-Level Analysis**: Analyze data at individual device, device type, floor, and building levels
- **Machine Learning Models**: Random Forest, Gradient Boosting, and LSTM neural networks
- **Advanced Anomaly Detection**: Statistical-based anomaly identification using trained models
- **Economic Feasibility Analysis**: Complete financial analysis with ROI, NPV, IRR calculations
- **Central Limit Theorem Analysis**: Statistical validation of data quality
- **Interactive Visualizations**: Dynamic charts and heatmaps for pattern exploration

### Technical Capabilities

- **Automated Feature Engineering**: Time-based features, lag variables, rolling averages with cyclical encoding
- **Multicollinearity Detection**: Automatic removal of highly correlated features (threshold > 0.7)
- **Time-Series Aware Splitting**: Chronological data splitting to prevent data leakage
- **Bulk Processing**: Train multiple models simultaneously with progress tracking
- **Aggregate Training**: Train models on combined datasets (buildings, device types, floors)
- **Data Export**: Download results as CSV files
- **Enhanced UI**: Improved contrast and responsive design for better accessibility

![Feature Engineering](correlation.png)

## 📊 Supported Device Types

| Device Type | Operating Hours | Description             |
| ----------- | --------------- | ----------------------- |
| AHU         | 08:00-16:00     | Air Handling Units      |
| SDP         | 00:00-23:00     | Sub Distribution Panels |
| LIFT        | 07:00-20:00     | Elevators               |
| CHILLER     | 08:00-17:00     | Cooling Systems         |

## 🛠 Installation

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-consumption-dashboard.git
cd energy-consumption-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Requirements.txt

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
seaborn>=0.12.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
plotly>=5.15.0
holidays>=0.32
scipy>=1.11.0
joblib>=1.3.0
```

## 📁 Data Structure

### Required Folder Structure

Your ZIP file should follow this structure:

```
data.zip
├── building_name/
│   ├── floor_name/
│   │   ├── device_type/
│   │   │   └── data.csv
│   │   └── another_device/
│   │       └── data.csv
│   └── another_floor/
└── another_building/
```

Alternative structure:

```
data.zip
├── building_name/
│   ├── device_type/
│   │   └── data.csv
│   └── another_device/
└── another_building/
```

### Required CSV Format

Your CSV files must contain:

- `id_time`: Timestamp column (will be used as index)
- `Konsumsi Energi`: Energy consumption values in Wh
- Additional weather/environmental columns (optional but recommended)

## 🎯 Usage Guide

### 1. Data Upload

1. Prepare your data in the required folder structure
2. Create a ZIP file containing all your data
3. Upload via the sidebar file uploader
4. Adjust minimum data points threshold if needed

![Upload Interface](image-1.png)

### 2. Data Overview

Navigate through the enhanced tab system to explore your data:

- **Data Overview**: Summary statistics and time series visualization with CLT analysis
- **Building Analysis**: Building-level consumption patterns and device distribution
- **Device Analysis (All Buildings)**: Device type comparison across all buildings
- **Device Analysis (Per Building)**: Device analysis within specific buildings
- **Floor Analysis**: Floor-level energy distribution

![Building Analysis](image-3.png)
![Building Analysis](image-4.png)
![Central Limit Theorem Analysis](image-2.png)

_Data overview tab showing consumption patterns and Central Limit Theorem analysis_

### 3. Enhanced Model Training

The training system now supports two comprehensive modes:

#### Individual Device Training

![Individual Device Training](image-5.png)

- Select specific devices for detailed analysis
- Enhanced feature engineering with cyclical time features
- Improved correlation analysis showing all features
- Comprehensive model performance metrics
- Detailed anomaly detection with 2σ threshold

#### Bulk Training System

![Bulk Training](image-7.png)
![Aggregate Training](image-8.png)

**Individual Bulk Training:**

- Train multiple devices simultaneously with progress tracking
- Organized device selection by building
- Comprehensive training summary with performance heatmaps
- Detailed results view for each trained device

**Aggregate Training Options:**

1. **All Data Combined**: Single model on entire dataset
2. **Per Building**: Separate model for each building
3. **Per Floor**: Model for each floor within buildings
4. **Per Device Type**: Model for each device type across buildings
5. **Per Device Type per Building**: Model for device types within specific buildings

![Model Performance Comparison](image-9.png)
![Model Performance Heatmap](image-10.png)
_Enhanced model training interface with heatmaps and aggregate options_

### 4. Advanced Anomaly Detection

The system provides comprehensive anomaly analysis:

- **Statistical Method**: Uses best ML model + 2σ threshold
- **Detailed Analysis**: Hour-by-hour and day-by-day anomaly patterns
- **Multi-level Analysis**: Overall, by building, device type, floor, or individual device
- **Scenario Analysis**: Separates work hours vs non-work hours anomalies
- **Potential Savings**: Quantifies energy waste and cost savings

![Anomaly Detection](image-11.png)
![Anomaly Pattern](image-12.png)
_Advanced anomaly detection with detailed pattern analysis_

### 5. Economic Analysis

Enhanced economic analysis with two calculation methods:

#### Model-Based Analysis

- Automatic calculation from anomaly detection results
- Annualized projections with seasonal considerations
- Comprehensive financial metrics

#### Manual Calculator

- Custom project parameter input
- Scenario comparison capabilities
- Independent of anomaly detection results

![Manual Calculator](image-13.png)
![Calculation Results](image-14.png)
_Economic analysis with enhanced visualization and manual calculator_

## 📊 Key Metrics Explained

### Model Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in Wh
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
- **R² Score**: Proportion of variance explained by the model (0-1, higher is better)

### Enhanced Feature Engineering

- **Cyclical Time Features**: Sine/cosine encoding for hour and day-of-week
- **Advanced Lag Features**: 1, 2, 3, 24, 48, and 168-hour lags
- **Rolling Statistics**: 3-hour and 24-hour moving averages
- **Holiday Integration**: Indonesia holiday calendar integration
- **Multicollinearity Removal**: Automatic correlation-based feature selection

### Economic Metrics

- **Payback Period**: Time to recover initial investment
- **ROI**: Return on Investment over project lifetime
- **NPV**: Net Present Value considering discount rate
- **IRR**: Internal Rate of Return

### Anomaly Detection

- **Statistical Threshold**: Predicted consumption + (2 × standard deviation)
- **Anomaly Rate**: Percentage of data points identified as anomalous
- **Savings Potential**: Sum of excess consumption above threshold

## 🧠 Enhanced Machine Learning Pipeline

### Advanced Feature Engineering

1. **Lag Features**: Previous 1, 2, 3, 24, 48, and 168 hours of consumption
2. **Rolling Averages**: 3-hour and 24-hour moving averages with lag-1 shift
3. **Cyclical Time Features**: Sine/cosine encoding for temporal patterns
4. **Holiday Detection**: Indonesia public holiday integration
5. **Weekend/Weekday Classification**: Business vs non-business hour analysis

### Improved Model Selection

The system trains three models and automatically selects based on lowest MAE:

- **Random Forest**: Ensemble method with feature importance analysis
- **Gradient Boosting**: Sequential ensemble for complex pattern capture
- **LSTM**: Neural network with standardized features for time series

### Enhanced Validation Strategy

- **Time-series aware splitting**: Prevents data leakage with chronological splits
- **Multicollinearity handling**: Removes features with correlation > 0.7
- **Comprehensive evaluation**: Multiple metrics for model assessment

![Data Preprocessing Pipeline](image-15.png)
![Modelling Pipeline](image-16.png)
_Enhanced machine learning pipeline with improved feature engineering_

## 🔬 Statistical Analysis

### Central Limit Theorem Validation

Enhanced CLT analysis with detailed visualization:

- **Population Distribution**: Original consumption data distribution
- **Sample Means Distribution**: CLT demonstration with n=30 samples
- **Statistical Comparison**: Theoretical vs actual normal distribution
- **Quality Assessment**: Automated evaluation of CLT conformity

This validates data suitability for statistical analysis and ML model reliability.

![CLT Analysis](image-18.png)
![CLT Analysis](image-19.png)
_Enhanced CLT visualization with quality assessment_

## 📈 Enhanced Visualization Features

### Interactive Charts

- **Comprehensive Time Series**: Multi-device overlay plots
- **Enhanced Correlation Matrices**: All features with dynamic sizing
- **Performance Heatmaps**: Model comparison across devices and aggregates
- **Anomaly Visualization**: Detailed threshold and pattern analysis
- **Economic Projections**: Interactive cash flow charts

### Export Capabilities

- **Enhanced CSV Downloads**: All analysis results exportable
- **Training Summaries**: Complete model performance reports
- **Economic Analysis**: Financial projection exports
- **Feature Analysis**: Correlation and importance data

## ⚡ Performance Considerations

### Data Requirements

- **Flexible minimum points**: 100-2000 data points per device (configurable)
- **Temporal resolution**: Hourly or sub-hourly timestamps recommended
- **Historical depth**: At least 30 days for reliable pattern detection

### Processing Optimizations

- **Individual device training**: ~30-60 seconds per device
- **Bulk training**: Efficient batch processing with progress tracking
- **Aggregate models**: Optimized for large combined datasets
- **Memory management**: Efficient pandas operations with caching

## 🛡 Data Privacy & Security

- **Local processing**: All analysis happens in browser/local environment
- **No external transmission**: Data never leaves your system
- **Session-based storage**: Temporary data handling only
- **Privacy by design**: No permanent data retention

## 🐛 Troubleshooting

### Common Issues

#### Enhanced Error Handling

- **Insufficient data warnings**: Clear minimum threshold messaging
- **ZIP structure validation**: Improved file structure checking
- **Feature engineering errors**: Better error messages for data quality issues

#### Training Issues

- **Memory optimization**: Improved handling for large datasets
- **Progress tracking**: Clear indication of training status
- **Model validation**: Enhanced error checking for model training

#### Performance Solutions

- **Batch processing**: Optimized bulk training algorithms
- **Memory management**: Efficient data structure usage
- **Progress feedback**: Real-time status updates

## 🆕 Recent Updates

### Version 2.0 Features

1. **Enhanced Training System**:

   - Bulk training for individual devices with building organization
   - Comprehensive aggregate training options
   - Progress tracking and status updates
   - Training summary with performance heatmaps

2. **Improved Feature Engineering**:

   - Cyclical encoding for temporal features
   - Extended lag feature set (up to 168 hours)
   - Advanced multicollinearity detection
   - Enhanced correlation visualization

3. **Advanced Anomaly Analysis**:

   - Multi-level analysis options
   - Scenario-based anomaly detection
   - Detailed pattern visualization
   - Work hours vs non-work hours analysis

4. **Enhanced User Interface**:

   - Improved contrast and accessibility
   - Responsive design for mobile devices
   - Better progress indicators
   - Enhanced data export options

5. **Statistical Validation**:
   - Central Limit Theorem analysis for all levels
   - Quality assessment automation
   - Enhanced statistical explanations

## 📚 Technical Documentation

### Enhanced Code Structure

```
├── app.py                 # Main Streamlit application with enhanced features
├── requirements.txt       # Updated Python dependencies
├── README.md             # This comprehensive documentation
└── data/                 # Example data structure
```

### Key Function Updates

- `engineer_features()`: Enhanced with cyclical encoding and extended lags
- `train_models()`: Improved with progress callbacks and better error handling
- `detect_anomalies_detailed()`: Comprehensive anomaly analysis with patterns
- `calculate_economic_metrics()`: Enhanced financial analysis
- `visualize_clt()`: Detailed CLT analysis with quality assessment
- `run_training_process()`: Unified training pipeline for all modes

## 🤝 Contributing

Contributions welcome! Recent focus areas:

1. **Performance Optimization**: Further improvements to bulk processing
2. **Additional ML Models**: Integration of more advanced algorithms
3. **Enhanced Visualizations**: More interactive and insightful charts
4. **Export Features**: Additional data export formats
5. **Documentation**: Expanded technical documentation

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run enhanced tests
python -m pytest tests/ -v

# Code formatting
black app.py
flake8 app.py --max-line-length=120
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for enhanced web interface
- Machine learning powered by [scikit-learn](https://scikit-learn.org/) and [TensorFlow](https://tensorflow.org/)
- Advanced visualizations with [Plotly](https://plotly.com/)
- Statistical analysis using [SciPy](https://scipy.org/)
- Holiday data from [python-holidays](https://python-holidays.readthedocs.io/)

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- GitHub Issues: [Create an issue](https://github.com/yourusername/energy-consumption-dashboard/issues)
- Email: your.email@domain.com
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

**Made with ❤️ for energy efficiency and sustainability**

### 🔄 Changelog

**Version 2.0** (Current):

- Major UI/UX improvements with better contrast and accessibility
- Enhanced bulk training system with individual and aggregate modes
- Advanced anomaly detection with multi-level analysis
- Improved feature engineering with cyclical encoding
- Comprehensive CLT analysis and statistical validation
- Enhanced economic analysis with manual calculator
- Better error handling and progress tracking
- Responsive design for mobile compatibility

**Version 1.0**:

- Initial release with basic ML training
- Simple anomaly detection
- Basic economic analysis
- Standard visualizations
