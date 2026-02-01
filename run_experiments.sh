export SAVE_DIR="./visualizations/comm=20-rep=5-high-discrete-non-extreme-ws"
# python3 new_models/plot_results.py --punishment none --save-dir $SAVE_DIR"/harsh_and_none" --device gpu

# python3 new_models/plot_results.py --punishment mild --save-dir $SAVE_DIR"/harsh_and_mild" --device gpu

# python3 new_models/plot_results.py --punishment total-punishment --save-dir $SAVE_DIR"/total-punishment" --device gpu

python3 new_models/plot_results.py --jsd --save-dir $SAVE_DIR"/jsd" --device gpu