export SAVE_DIR="./visualizations/w-1/changed-u-target-extreme-ws"
python3 new_models/plot_results.py --punishment none --save-dir $SAVE_DIR"/harsh_and_none" --device gpu

python3 new_models/plot_results.py --punishment mild --save-dir $SAVE_DIR"/harsh_and_mild" --device gpu

python3 new_models/plot_results.py --punishment total-punishment --save-dir $SAVE_DIR"/total-punishment" --device gpu

python3 new_models/plot_results.py --belief-update --save-dir $SAVE_DIR"/belief-update" --device gpu

python3 new_models/plot_results.py --utility-mode --save-dir $SAVE_DIR"/utility-all" --device gpu

python3 new_models/plot_results.py --w1 --save-dir $SAVE_DIR"/w1" --device gpu