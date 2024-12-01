function ErrorMessage({ message, onClose }) {
  return (
    <div className="error-message">
      <div className="error-content">
        <span className="error-icon">⚠️</span>
        <p>{message}</p>
        <button onClick={onClose}>×</button>
      </div>
    </div>
  );
} 