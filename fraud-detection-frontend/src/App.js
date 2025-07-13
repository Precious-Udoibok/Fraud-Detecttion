import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    id: '',
    Amount: '',
    V1: '',
    V2: '',
    V3: '',
    V4: '',
    V5: '',
    V6: '',
    V7: '',
    V8: '',
    V9: '',
    V10: '',
    V11: '',
    V12: '',
    V13: '',
    V14: '',
    V15: '',
    V16: '',
    V17: '',
    V18: '',
    V19: '',
    V20: '',
    V21: '',
    V22: '',
    V23: '',
    V24: '',
    V25: '',
    V26: '',
    V27: '',
    V28: ''
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    // Allow empty or valid number input (including negative)
    if (value === '' || value === '-' || !isNaN(value)) {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const validateForm = () => {
    // Check if all fields are valid numbers
    for (const key in formData) {
      if (formData[key] !== '' && isNaN(formData[key])) {
        setError(`Invalid value for ${key}. Please enter a valid number.`);
        return false;
      }
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    // Convert all values to numbers before sending
    const numericFormData = {
      id: parseFloat(formData.id) || 0,
      Amount: parseFloat(formData.Amount) || 0,
      V1: parseFloat(formData.V1) || 0,
      V2: parseFloat(formData.V2) || 0,
      V3: parseFloat(formData.V3) || 0,
      V4: parseFloat(formData.V4) || 0,
      V5: parseFloat(formData.V5) || 0,
      V6: parseFloat(formData.V6) || 0,
      V7: parseFloat(formData.V7) || 0,
      V8: parseFloat(formData.V8) || 0,
      V9: parseFloat(formData.V9) || 0,
      V10: parseFloat(formData.V10) || 0,
      V11: parseFloat(formData.V11) || 0,
      V12: parseFloat(formData.V12) || 0,
      V13: parseFloat(formData.V13) || 0,
      V14: parseFloat(formData.V14) || 0,
      V15: parseFloat(formData.V15) || 0,
      V16: parseFloat(formData.V16) || 0,
      V17: parseFloat(formData.V17) || 0,
      V18: parseFloat(formData.V18) || 0,
      V19: parseFloat(formData.V19) || 0,
      V20: parseFloat(formData.V20) || 0,
      V21: parseFloat(formData.V21) || 0,
      V22: parseFloat(formData.V22) || 0,
      V23: parseFloat(formData.V23) || 0,
      V24: parseFloat(formData.V24) || 0,
      V25: parseFloat(formData.V25) || 0,
      V26: parseFloat(formData.V26) || 0,
      V27: parseFloat(formData.V27) || 0,
      V28: parseFloat(formData.V28) || 0,
    };

    console.log('Sending data to API:', numericFormData); // Log the data

    try {
      const response = await axios.post('https://fraud-detecttion.onrender.com/predict', numericFormData, {
        headers: {
          'Content-Type': 'application/json', // Ensure the correct header is sent
        },
      });
      setResult(response.data);
    } catch (err) {
      console.error('API Error:', err.response ? err.response.data : err.message); // Log the error
      setError('Error making prediction. Please check your inputs and try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      id: '',
      Amount: '',
      V1: '',
      V2: '',
      V3: '',
      V4: '',
      V5: '',
      V6: '',
      V7: '',
      V8: '',
      V9: '',
      V10: '',
      V11: '',
      V12: '',
      V13: '',
      V14: '',
      V15: '',
      V16: '',
      V17: '',
      V18: '',
      V19: '',
      V20: '',
      V21: '',
      V22: '',
      V23: '',
      V24: '',
      V25: '',
      V26: '',
      V27: '',
      V28: ''
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="background"></div>
      <div className="container">
        <div className="header">
          <h1>Credit Card Fraud Detection</h1>
          <p>Advanced machine learning system for real-time fraud detection</p>
        </div>

        <form onSubmit={handleSubmit} className="form">
          <div className="primary-fields">
            <div className="input-group">
              <label>Transaction ID:</label>
              <input
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                name="id"
                value={formData.id}
                onChange={handleInputChange}
                placeholder="Enter ID"
                required
              />
            </div>
            <div className="input-group">
              <label>Amount:</label>
              <input
                type="text"
                inputMode="decimal"
                name="Amount"
                value={formData.Amount}
                onChange={handleInputChange}
                placeholder="Enter amount"
                required
              />
            </div>
          </div>

          <div className="features-section">
            <h2>Transaction Features</h2>
            <div className="input-grid">
              {Object.keys(formData).map((field) => {
                if (field !== 'id' && field !== 'Amount') {
                  return (
                    <div key={field} className="input-group">
                      <label>{field}:</label>
                      <input
                        type="text"
                        inputMode="decimal"
                        name={field}
                        value={formData[field]}
                        onChange={handleInputChange}
                        placeholder="Enter value"
                        required
                      />
                    </div>
                  );
                }
                return null;
              })}
            </div>
          </div>

          <div className="button-group">
            <button 
              type="submit" 
              className="predict-button"
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Transaction'}
            </button>
            <button 
              type="button" 
              className="reset-button"
              onClick={resetForm}
            >
              Reset Form
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}
        
        {result && (
          <div className={`result ${result.is_fraud ? 'fraud' : 'not-fraud'}`}>
            <h2>
              {result.is_fraud ? 'Fraudulent Transaction Detected' : 'Valid Transaction'}
            </h2>
            <p className="probability">
              Fraud Probability: {(result.probability * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
