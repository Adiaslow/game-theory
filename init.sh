echo
echo "Creating virtual environment..."
python3.12 -m venv game_theory_env
echo
echo "Activating virtual environment..."
source game_theory_env/bin/activate
echo
echo "Installing dependencies..."
pip install joblib
pip install matplotlib
pip install scikit-learn
pip install tqdm
echo
pip list
echo
echo "Done!"
echo
echo "To deactivate the virtual environment, run 'sh term.sh'"
echo "Run 'python3.12 game_directory/game.py' to run the game."
echo
