export SAVE_DIR="./visualizations/final_plots/run_5_rep_and_comm_w_cost_self_0.05"
# python3 new_models/plot_results.py --punishment none --save-dir $SAVE_DIR"/harsh_and_none" --device gpu

# python3 new_models/plot_results.py --punishment mild --save-dir $SAVE_DIR"/harsh_and_mild" --device gpu

# python3 new_models/plot_results.py --punishment total-punishment --save-dir $SAVE_DIR"/total-punishment" --device gpu

# python3 new_models/plot_results.py --belief-update --save-dir $SAVE_DIR"/belief-update" --device gpu

# python3 new_models/plot_results.py --utility-mode --punishment all --save-dir $SAVE_DIR"/utility-all" --device gpu

# python3 new_models/plot_results.py --w1 --save-dir $SAVE_DIR"/w1" --device gpu

python3 new_models/plot_results.py --final-plots --save-dir $SAVE_DIR --device gpu