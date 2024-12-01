import os
from app import app
from config.production import ProductionConfig

# Load production config
app.config.from_object(ProductionConfig)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port) 