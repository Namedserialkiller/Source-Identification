# OpenAQ API Integration

This project supports real-time air quality data fetching from the OpenAQ API for Delhi pollution source identification.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAQ API Key:**
   - Sign up at https://openaq.org/
   - Get your API key from the dashboard

3. **Configure API Key:**
   - Create a `.env` file in the project root:
   ```
   OPENAQ_API_KEY=your_actual_api_key_here
   ```
   - **Important:** Never commit the `.env` file to version control (it's in `.gitignore`)

## Usage

### Training with Live Data

Train the model using live data from OpenAQ:
```bash
cd src
python main.py --live
```

### Real-time Prediction

Predict pollution source from latest Delhi AQI data:
```bash
cd src
python predict_live.py
```

With custom options:
```bash
python predict_live.py --hours-back 6  # Look back 6 hours
python predict_live.py --no-probabilities  # Don't show probabilities
```

### Using CSV Data (Fallback)

If API is unavailable or for historical data:
```bash
python main.py "path/to/data.csv"
```

## Error Handling

The system handles several error scenarios:

1. **Missing API Key:** Falls back to CSV if provided, or raises clear error
2. **401 Unauthorized:** Raises `UnauthorizedError` with instructions
3. **Missing Parameters:** Fills with historical Delhi means (from training dataset)
4. **API Timeout/Failure:** Falls back to historical means for prediction

## Data Format

The OpenAQ API data is automatically mapped to the expected format:
- `pm25` → `pm2_5`
- `pm10` → `pm10`
- `co`, `no2`, `o3`, `so2`, `nh3`, `no` → same names

Missing parameters are filled with historical means from the Delhi dataset.

## Notes

- The API queries Delhi area (coordinates: 28.6139, 77.2090) with 50km radius
- Default lookback period: 24 hours for training, 1 hour for prediction
- Timestamps are preserved for month-based heuristics (stubble burning detection)
