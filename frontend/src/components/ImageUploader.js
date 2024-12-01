import React, { useState } from 'react';

const ImageUploader = ({ onImageUpload, label }) => {
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const validateImage = (file) => {
    const validTypes = ['image/jpeg', 'image/png'];
    const maxSize = 5 * 1024 * 1024; // 5 MB

    if (!validTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload a JPEG or PNG image.');
    }
    if (file.size > maxSize) {
      throw new Error('File size exceeds 5 MB.');
    }
    return true;
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      setError(null);
      setLoading(true);
      validateImage(file);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);

      // Upload image
      const formData = new FormData();
      formData.append('image', file);
      await onImageUpload(formData);
    } catch (err) {
      setError(err.message);
      e.target.value = ''; // Reset input
      setPreview(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`image-uploader ${loading ? 'loading' : ''}`}>
      <label>
        {label}
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleFileChange}
          disabled={loading}
        />
      </label>
      {loading && <div className="loader">Uploading...</div>}
      {error && <div className="error">{error}</div>}
      {preview && (
        <img 
          src={preview} 
          alt="Preview" 
          style={{ maxWidth: '200px', marginTop: '10px' }}
        />
      )}
    </div>
  );
};

export default ImageUploader; 