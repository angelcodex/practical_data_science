#!/bin/bash
# Setup script for ExtraaLearn Potential Customer Prediction Project

PROJECT_NAME="extraalearn"

echo "📚 ExtraaLearn Potential Customer Prediction - Project Setup"
echo "============================================================"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "🔌 Registering Jupyter kernel..."
python -m ipykernel install --user --name=${PROJECT_NAME}_venv --display-name="Python (${PROJECT_NAME})"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate venv: source venv/bin/activate"
echo "   2. Open your notebook in notebooks/ directory"
echo "   3. Select kernel: 'Python (extraalearn)'"
echo "   4. Restart the kernel after selecting"
echo "   5. Update data path in notebook to: '../data/ExtraaLearn.csv'"
echo ""


