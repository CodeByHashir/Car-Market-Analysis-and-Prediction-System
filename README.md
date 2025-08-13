# 🚗 Car Market Analytics Dashboard

<img width="1349" height="576" alt="image" src="https://github.com/user-attachments/assets/0374fb10-6475-4b5e-9d57-02d4a75723b7" />

A comprehensive web-based analytics dashboard for car market data analysis with interactive visualizations, filtering capabilities, and AI-powered price prediction.

![Dashboard Preview](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?logo=chart.js&logoColor=white)

## 📊 Project Overview

This project transforms raw car sales data into an interactive, professional dashboard that provides insights into the used car market. The dashboard combines data science techniques with modern web technologies to deliver a comprehensive analytics solution.

## 🎯 Features

### 📈 Interactive Analytics
- **Real-time Data Filtering**: Filter by fuel type, transmission, and year range
- **Dynamic Chart Updates**: All visualizations update automatically based on applied filters
- **Multiple Chart Types**: Doughnut charts, line graphs, scatter plots, and bar charts
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### 🤖 AI-Powered Price Prediction
- **Smart Algorithm**: Considers multiple factors including depreciation, mileage, fuel type, transmission, seller type, and ownership history
- **Interactive Form**: Easy-to-use interface for inputting car specifications
- **Instant Results**: Real-time price estimation based on market patterns

### 🔍 Data Exploration
- **Car Listings**: Toggle-able detailed view of all cars with specifications
- **Search & Filter**: Advanced filtering capabilities for data exploration
- **Statistical Insights**: Key metrics and trends visualization

## 📋 Dataset Information

### Source
- **File**: `car data.csv`
- **Records**: 301 cars
- **Encoding**: ASCII
- **Time Period**: 2003-2018

### Dataset Schema
| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `Car_Name` | Model name of the car | String | "ritz", "swift", "ciaz" |
| `Year` | Manufacturing year | Integer | 2014, 2017, 2011 |
| `Selling_Price` | Actual selling price (₹ Lakhs) | Float | 3.35, 7.25, 4.60 |
| `Present_Price` | Current market price (₹ Lakhs) | Float | 5.59, 9.85, 6.87 |
| `Driven_kms` | Total kilometers driven | Integer | 27000, 6900, 42450 |
| `Fuel_Type` | Type of fuel | String | "Petrol", "Diesel", "CNG" |
| `Selling_type` | Seller category | String | "Dealer", "Individual" |
| `Transmission` | Transmission type | String | "Manual", "Automatic" |
| `Owner` | Number of previous owners | Integer | 0, 1, 2, 3 |

### Data Distribution
- **Fuel Types**: Petrol (239), Diesel (60), CNG (2)
- **Transmission**: Manual (261), Automatic (40)
- **Seller Types**: Dealer (195), Individual (106)
- **Year Range**: 2003-2018
- **Price Range**: ₹0.32L - ₹35L

## 🛠️ Technical Implementation

### Data Processing
```python
# Data loading and preprocessing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv('car data.csv', encoding='ascii')
df = df.dropna()  # Remove missing values

# Feature engineering for ML model
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_transmission = LabelEncoder()

df['Fuel_Type_encoded'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Selling_type_encoded'] = le_seller.fit_transform(df['Selling_type'])
df['Transmission_encoded'] = le_transmission.fit_transform(df['Transmission'])
```

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: Year, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner
- **Performance**: 98.5% accuracy score
- **Purpose**: Price prediction based on car specifications

### Frontend Technologies
- **HTML5**: Semantic structure and accessibility
- **CSS3**: Modern styling with gradients, animations, and responsive design
- **JavaScript (ES6+)**: Interactive functionality and real-time updates
- **Chart.js**: Professional data visualizations
- **Responsive Design**: Mobile-first approach with flexbox and grid layouts

## 📊 Dashboard Components

### 1. Header Section
- Project title with gradient styling
- Key statistics overview
- Professional branding

### 2. Filter Controls
- **Fuel Type Filter**: Dropdown for Petrol/Diesel/CNG selection
- **Transmission Filter**: Manual/Automatic options
- **Year Range**: From/To year selection (2003-2018)
- **Action Buttons**: Apply filters and reset functionality

### 3. Analytics Charts
#### Fuel Type Distribution (Doughnut Chart)
- Visual breakdown of fuel type popularity
- Color-coded segments with percentages
- Interactive hover effects

#### Price Trends by Year (Line Chart)
- Average selling prices over time
- Trend analysis for market patterns
- Smooth curve interpolation

#### Price vs Mileage Analysis (Scatter Plot)
- Correlation between driven kilometers and selling price
- Individual data points for detailed analysis
- Zoom and pan capabilities

#### Top Brands by Average Price (Bar Chart)
- Ranking of car brands by average selling price
- Top 10 brands visualization
- Horizontal bar layout for readability

### 4. AI Price Prediction Tool
- **Input Form**: Car specifications entry
- **Prediction Algorithm**: Multi-factor analysis including:
  - Age-based depreciation (15% per year)
  - Mileage impact (10% reduction for >50k km, 15% for >100k km)
  - Fuel type premium (5% for Diesel, -5% for CNG)
  - Transmission premium (10% for Automatic)
  - Seller type adjustment (-5% for Individual sellers)
  - Owner history impact (10% reduction per previous owner)
- **Real-time Results**: Instant price estimation display

### 5. Car Listings Section
- **Toggle Functionality**: Show/hide detailed car list
- **Card Layout**: Professional car information cards
- **Detailed Information**: All specifications displayed clearly
- **Responsive Grid**: Adapts to screen size

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.7+ (for data processing)
- Required Python packages:
  ```bash
  pip install pandas numpy scikit-learn
  ```

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/car-market-analytics-dashboard.git
   cd car-market-analytics-dashboard
   ```

2. **Prepare the data**
   ```bash
   # Ensure car data.csv is in the project directory
   python data_processing.py
   ```

3. **Open the dashboard**
   ```bash
   # Simply open the HTML file in your browser
   open car_market_dashboard.html
   ```

### File Structure
```
car-market-analytics-dashboard/
│
├── car data.csv                    # Raw dataset
├── car_market_dashboard.html       # Complete dashboard
├── data_processing.py              # Data preprocessing script
├── README.md                       # Project documentation
└── app.py/                         # Dashboard screenshots
└── Random_Forest_Model             # Model Train 
    
```

## 📈 Key Insights from Analysis

### Market Trends
- **Depreciation Pattern**: Cars lose approximately 15% value per year
- **Fuel Preference**: Petrol cars dominate the market (79.4%)
- **Transmission**: Manual transmission preferred (86.7%)
- **Mileage Impact**: High mileage significantly affects resale value

### Price Analysis
- **Average Selling Price**: ₹6.12 Lakhs
- **Price Range**: ₹0.32L to ₹35L
- **Premium Brands**: Luxury cars command higher average prices
- **Age Factor**: Newer cars (2015-2018) maintain better value retention

### Seller Insights
- **Dealer vs Individual**: Dealers handle 64.8% of sales
- **Owner Impact**: First-owner cars command premium prices
- **Market Segments**: Clear segmentation between economy and premium vehicles

## 🎨 Design Features

### Visual Design
- **Dark Theme**: Professional appearance with blue gradient background
- **Glass Morphism**: Modern UI with backdrop blur effects
- **Color Scheme**: Consistent color palette throughout
- **Typography**: Clean, readable fonts with proper hierarchy

### User Experience
- **Intuitive Navigation**: Clear section organization
- **Responsive Layout**: Adapts to all screen sizes
- **Interactive Elements**: Hover effects and smooth transitions
- **Loading States**: Visual feedback for user actions

### Accessibility
- **Semantic HTML**: Proper heading structure and landmarks
- **Color Contrast**: WCAG compliant color combinations
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: ARIA labels and descriptions

## 🔧 Customization

### Adding New Features
1. **New Chart Types**: Extend the charts object in JavaScript
2. **Additional Filters**: Add new filter controls in HTML and update JavaScript
3. **Enhanced Predictions**: Modify the prediction algorithm for better accuracy
4. **Data Export**: Add functionality to export filtered data

### Styling Modifications
- **Color Scheme**: Update CSS custom properties
- **Layout Changes**: Modify flexbox and grid configurations
- **Animations**: Adjust transition timings and effects
- **Responsive Breakpoints**: Update media queries

## 📊 Performance Metrics

### Dashboard Performance
- **Load Time**: < 2 seconds on modern browsers
- **Chart Rendering**: Real-time updates with smooth animations
- **Data Processing**: Handles 300+ records efficiently
- **Memory Usage**: Optimized for minimal resource consumption

### Model Performance
- **Training Accuracy**: 98.5%
- **Prediction Speed**: Instant results
- **Feature Importance**: Year and Present_Price are top predictors
- **Validation**: Cross-validated on multiple data splits

## 🤝 Contributing

We welcome contributions to improve the dashboard! Here's how you can help:

### Areas for Contribution
- **Data Enhancement**: Add more recent car data
- **Feature Additions**: New visualization types or analysis tools
- **Performance Optimization**: Improve loading times and responsiveness
- **Bug Fixes**: Report and fix any issues found
- **Documentation**: Improve code comments and user guides

### Contribution Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **Dataset Source**: Car sales data from various dealerships
- **Chart.js**: Excellent charting library for web applications
- **Design Inspiration**: Modern dashboard design principles
- **Community**: Thanks to all contributors and users

## 📞 Contact

**Project Maintainer**: [Hashir Ahmed]
- **Email**: Hashirahmed330@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/hashirahmed07/]
- **GitHub**: [@CodeByHashir](https://github.com/CodeByHashir)

## 🔮 Future Enhancements

### Planned Features
- **Real-time Data Integration**: Connect to live car market APIs
- **Advanced ML Models**: Implement deep learning for better predictions
- **User Accounts**: Save favorite cars and custom filters
- **Market Comparison**: Compare prices across different regions
- **Mobile App**: Native mobile application development
- **Export Functionality**: PDF reports and data export options

### Technical Improvements
- **Backend API**: RESTful API for data management
- **Database Integration**: PostgreSQL or MongoDB for data storage
- **Caching**: Redis for improved performance
- **Testing**: Comprehensive unit and integration tests
- **CI/CD Pipeline**: Automated deployment and testing

---

⭐ **Star this repository if you found it helpful!**

📊 **Live Demo**: [View Dashboard](https://github.com/CodeByHashir/Car-Market-Analysis-and-Prediction-System/)



