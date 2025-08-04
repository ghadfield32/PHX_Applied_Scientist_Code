import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

const NBAContractPredictor = () => {
  const [playerStats, setPlayerStats] = useState({
    GP: '',
    GS: '',
    MP: '',
    FG: '',
    FGA: '',
    'FG_PCT': '',
    '3P': '',
    '3PA': '',
    '3P%': '',
    '2P': '',
    '2PA': '',
    '2P%': '',
    FT: '',
    FTA: '',
    'FT%': '',
    PER: '',
    BPM: '',
    OBPM: '',
    DBPM: '',
    VORP: '',
    WS: '',
    'WS/48': '',
    'USG%': '',
    'TS%': '',
    'EFG%': '',
    TRB: '',
    AST: '',
    STL: '',
    BLK: '',
    TOV: '',
    PF: '',
    PTS: '',
    W: '',
    L: '',
    W_PCT: '',
    AGE: '',
    SEASON: '2023-24',
    POSITION: 'Guard',
    experience_bucket: '3-5',
    portability_score: '',
    TOTAL_DAYS_INJURED: '0'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    checkApiHealth();
    fetchModelInfo();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setApiHealth(data);
    } catch (err) {
      setApiHealth({ status: 'unhealthy', error: err.message });
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model/info`);
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  };

  const handleInputChange = (field, value) => {
    setPlayerStats(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      // Convert string inputs to numbers where needed
      const numericStats = {};
      Object.keys(playerStats).forEach(key => {
        const value = playerStats[key];
        if (['SEASON', 'POSITION', 'experience_bucket'].includes(key)) {
          numericStats[key] = value;
        } else if (value === '') {
          numericStats[key] = 0;
        } else {
          numericStats[key] = parseFloat(value) || 0;
        }
      });

      const response = await fetch(`${API_BASE_URL}/predict?include_confidence=true`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(numericStats)
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExamplePlayer = () => {
    setPlayerStats({
      GP: '65',
      GS: '60', 
      MP: '2000',
      FG: '400',
      FGA: '900',
      'FG_PCT': '0.444',
      '3P': '120',
      '3PA': '350',
      '3P%': '0.343',
      '2P': '280',
      '2PA': '550',
      '2P%': '0.509',
      FT: '180',
      FTA: '200',
      'FT%': '0.900',
      PER: '18.5',
      BPM: '4.2',
      OBPM: '3.8',
      DBPM: '0.4',
      VORP: '3.5',
      WS: '8.2',
      'WS/48': '0.150',
      'USG%': '24.5',
      'TS%': '0.580',
      'EFG%': '0.511',
      TRB: '280',
      AST: '350',
      STL: '85',
      BLK: '45',
      TOV: '180',
      PF: '150',
      PTS: '1100',
      W: '45',
      L: '37',
      W_PCT: '0.549',
      AGE: '27',
      SEASON: '2023-24',
      POSITION: 'Guard',
      experience_bucket: '6-9',
      portability_score: '0.65',
      TOTAL_DAYS_INJURED: '15'
    });
  };

  const formatPercentage = (value) => {
    return (value * 100).toFixed(2) + '%';
  };

  const formatCurrency = (aavPct, capAmount = 136000000) => {
    const aavAmount = aavPct * capAmount;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(aavAmount);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">NBA Contract Predictor</h1>
        <p className="text-gray-600">Predict player market value using advanced analytics</p>
      </div>

      {/* API Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${apiHealth?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}></div>
            API Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600">Status</p>
              <p className="font-semibold">{apiHealth?.status || 'Unknown'}</p>
            </div>
            {modelInfo && (
              <>
                <div>
                  <p className="text-sm text-gray-600">Model</p>
                  <p className="font-semibold">{modelInfo.model_name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Features</p>
                  <p className="font-semibold">{modelInfo.feature_count}</p>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Player Statistics Input
              <button
                onClick={loadExamplePlayer}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
              >
                Load Example
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Basic Info */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Age</label>
                <input
                  type="number"
                  value={playerStats.AGE}
                  onChange={(e) => handleInputChange('AGE', e.target.value)}
                  className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Position</label>
                <select
                  value={playerStats.POSITION}
                  onChange={(e) => handleInputChange('POSITION', e.target.value)}
                  className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="Guard">Guard</option>
                  <option value="Forward">Forward</option>
                  <option value="Center">Center</option>
                  <option value="Guard-Forward">Guard-Forward</option>
                  <option value="Forward-Guard">Forward-Guard</option>
                  <option value="Center-Forward">Center-Forward</option>
                  <option value="Forward-Center">Forward-Center</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Experience</label>
                <select
                  value={playerStats.experience_bucket}
                  onChange={(e) => handleInputChange('experience_bucket', e.target.value)}
                  className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="0">Rookie</option>
                  <option value="1-2">1-2 Years</option>
                  <option value="3-5">3-5 Years</option>
                  <option value="6-9">6-9 Years</option>
                  <option value="10-14">10-14 Years</option>
                  <option value="15+">15+ Years</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Season</label>
                <input
                  type="text"
                  value={playerStats.SEASON}
                  onChange={(e) => handleInputChange('SEASON', e.target.value)}
                  placeholder="2023-24"
                  className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* Key Stats */}
            <div className="space-y-3">
              <h4 className="font-medium text-gray-900">Key Statistics</h4>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { key: 'GP', label: 'Games Played', max: 82 },
                  { key: 'GS', label: 'Games Started', max: 82 },
                  { key: 'MP', label: 'Minutes Played' },
                  { key: 'PTS', label: 'Points' },
                  { key: 'TRB', label: 'Rebounds' },
                  { key: 'AST', label: 'Assists' },
                  { key: 'FG_PCT', label: 'FG%', step: 0.001, max: 1 },
                  { key: '3P%', label: '3P%', step: 0.001, max: 1 }
                ].map(({ key, label, max, step }) => (
                  <div key={key}>
                    <label className="block text-sm font-medium mb-1">{label}</label>
                    <input
                      type="number"
                      value={playerStats[key]}
                      onChange={(e) => handleInputChange(key, e.target.value)}
                      max={max}
                      step={step || 1}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Advanced Metrics */}
            <div className="space-y-3">
              <h4 className="font-medium text-gray-900">Advanced Metrics</h4>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { key: 'PER', label: 'PER' },
                  { key: 'BPM', label: 'BPM' },
                  { key: 'VORP', label: 'VORP' },
                  { key: 'WS', label: 'Win Shares' },
                  { key: 'USG%', label: 'Usage %' },
                  { key: 'TS%', label: 'True Shooting %', step: 0.001, max: 1 }
                ].map(({ key, label, max, step }) => (
                  <div key={key}>
                    <label className="block text-sm font-medium mb-1">{label}</label>
                    <input
                      type="number"
                      value={playerStats[key]}
                      onChange={(e) => handleInputChange(key, e.target.value)}
                      max={max}
                      step={step || 0.1}
                      className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || apiHealth?.status !== 'healthy'}
              className="w-full py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Predicting...' : 'Predict Contract Value'}
            </button>
          </CardContent>
        </Card>

        {/* Results */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
          </CardHeader>
          <CardContent>
            {error && (
              <Alert className="mb-4">
                <AlertDescription className="text-red-600">
                  Error: {error}
                </AlertDescription>
              </Alert>
            )}

            {prediction && (
              <div className="space-y-4">
                <div className="text-center p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Predicted AAV</h3>
                  <div className="text-3xl font-bold text-blue-600">
                    {formatPercentage(prediction.predicted_aav_pct_cap)}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">of salary cap</div>
                  <div className="text-xl font-semibold text-gray-800 mt-2">
                    ≈ {formatCurrency(prediction.predicted_aav_pct_cap)}
                  </div>
                </div>

                {prediction.confidence_interval_lower && (
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Confidence Interval (95%)</h4>
                    <div className="flex justify-between text-sm">
                      <span>Lower: {formatPercentage(prediction.confidence_interval_lower)}</span>
                      <span>Upper: {formatPercentage(prediction.confidence_interval_upper)}</span>
                    </div>
                  </div>
                )}

                {prediction.risk_factors && prediction.risk_factors.length > 0 && (
                  <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                    <h4 className="font-medium text-yellow-800 mb-2">Risk Factors</h4>
                    <ul className="text-sm text-yellow-700 space-y-1">
                      {prediction.risk_factors.map((factor, index) => (
                        <li key={index}>• {factor}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="text-xs text-gray-500 space-y-1">
                  <div>Model: {prediction.model_version}</div>
                  <div>Predicted: {new Date(prediction.prediction_timestamp).toLocaleString()}</div>
                </div>
              </div>
            )}

            {!prediction && !error && (
              <div className="text-center py-8 text-gray-500">
                <p>Enter player statistics and click "Predict Contract Value" to see results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model Performance */}
      {modelInfo && (
        <Card>
          <CardHeader>
            <CardTitle>Model Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-600">Training Samples</p>
                <p className="text-xl font-semibold">{modelInfo.training_samples?.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">R² Score</p>
                <p className="text-xl font-semibold">{modelInfo.validation_metrics?.r2?.toFixed(3)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">MAE</p>
                <p className="text-xl font-semibold">{modelInfo.validation_metrics?.mae?.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">RMSE</p>
                <p className="text-xl font-semibold">{modelInfo.validation_metrics?.rmse?.toFixed(4)}</p>
              </div>
            </div>

            {modelInfo.top_features && (
              <div className="mt-4">
                <h4 className="font-medium mb-2">Top Features</h4>
                <div className="space-y-2">
                  {modelInfo.top_features.map((feature, index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm">{feature.feature}</span>
                      <span className="text-sm font-medium">{(feature.importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default NBAContractPredictor;
