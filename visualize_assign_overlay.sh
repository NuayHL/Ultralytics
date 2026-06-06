# python visualize_assign_overlay.py \
#     --baseline_ckpt runs/detect/aitodv2/v12s/weights/best.pt \
#     --ours_ckpt     runs/detect/aitodv2/yolo12s_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml/weights/best.pt \
#     --image         Img_aitod/12389.png \
#     --labels        Img_aitod/12389.txt \
#     --crop 0 0 640 640 \
#     --ours_assigner_kwargs '{"dyab_type":"DyabCalibrationAware","dyab_kwargs":{"alpha_base":1.0,"beta_base":4.0,"delta_alpha":0.5,"delta_beta":2.0,"r_ref":64.0},"r_ref":32,"r_ref_type":"add_1","dscale_func":"static","scale_ratio":1.0}' \
#     --out assign_visualization/comparison.png
#     # --class_colors None \



# python visualize_assign_overlay.py \
#     --baseline_ckpt runs/detect/visdrone/v12s/weights/best.pt \
#     --ours_ckpt     runs/detect/visdrone/yolo12s_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml/weights/best.pt \
#     --image         Img_visdrone/0000153_00401_d_0000001.jpg \
#     --labels        Img_visdrone/0000153_00401_d_0000001.txt \
#     --crop 0 0 640 640 \
#     --ours_assigner_kwargs '{"dyab_type":"DyabCalibrationAware","dyab_kwargs":{"alpha_base":1.0,"beta_base":4.0,"delta_alpha":0.5,"delta_beta":2.0,"r_ref":64.0},"r_ref":32,"r_ref_type":"add_1","dscale_func":"static","scale_ratio":1.0}' \
#     --out assign_visualization/comparison_visdrone_green.png \
#     --level P3 \
#     --gt_lw 0.5 \
#     --grid_color "#1B1C3F" \
#     # --class_colors None \

for img in Img_visdrone_1/*.{jpg,png}; do
    [ -e "$img" ] || continue

    base=$(basename "$img")
    base="${base%.*}"

    python visualize_assign_overlay.py \
        --baseline_ckpt runs/detect/visdrone/v12s/weights/best.pt \
        --ours_ckpt runs/detect/visdrone/yolo12s_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml/weights/best.pt \
        --image "$img" \
        --labels "Img_visdrone_1/${base}.txt" \
        --crop 160 160 480 480 \
        --ours_assigner_kwargs '{"dyab_type":"DyabCalibrationAware","dyab_kwargs":{"alpha_base":1.0,"beta_base":4.0,"delta_alpha":0.5,"delta_beta":2.0,"r_ref":64.0},"r_ref":32,"r_ref_type":"add_1","dscale_func":"static","scale_ratio":1.0}' \
        --out "assign_visualization_1/${base}.png" \
        --level P3 \
        --gt_lw 0.5 \
        --grid_color "#1B1C3F"
done
