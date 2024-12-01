import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import './App.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageUpload = async (file, type) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('image', file);
      
      const response = await fetch(`/upload-${type}-image`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error.message || 'Upload failed');
      }
      
      return data.data.filename;
      
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const processImages = async (personImage, garmentImage) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/process-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          person_image: personImage,
          garment_image: garmentImage
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error.message || 'Processing failed');
      }
      
      setResult(data.data.result);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <ErrorBoundary>
        <header>
          <h1>Virtual Try-On</h1>
        </header>

        {error && (
          <ErrorMessage 
            message={error}
            onClose={() => setError(null)}
          />
        )}

        <div className="upload-section">
          <ImageUploader
            label="Person Image"
            onUpload={(file) => handleImageUpload(file, 'person')}
            loading={loading}
            accept="image/jpeg,image/png"
            maxSize={5 * 1024 * 1024}
          />
          
          <ImageUploader
            label="Garment Image"
            onUpload={(file) => handleImageUpload(file, 'garment')}
            loading={loading}
            accept="image/jpeg,image/png"
            maxSize={5 * 1024 * 1024}
          />
        </div>

        <ProcessButton
          onClick={processImages}
          disabled={loading || !personImage || !garmentImage}
          loading={loading}
        />

        {result && (
          <ResultDisplay
            imageData={result}
            onClose={() => setResult(null)}
          />
        )}

        {loading && <LoadingSpinner />}
      </ErrorBoundary>
    </div>
  );
}

export default App; 