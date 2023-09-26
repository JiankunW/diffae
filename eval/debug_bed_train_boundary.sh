BOUNDARY_NAME=indoor_lighting
OUTPUT_DIR="../runs/bedroom128/pdae64_stopgrad/features_save-540k_first500000_ema"
SCORES_DIR="../runs/bedroom128/score_first500000"
python bed_train_boundary.py $OUTPUT_DIR/feats.npy $SCORES_DIR/attribute.npy \
    --score_name=$BOUNDARY_NAME \
    --output_dir=$OUTPUT_DIR \
    --gpu 0 \
    --epochs 10 \
    --promask 16 \
    --bs 32
