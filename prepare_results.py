import os
import re
import shutil

results_dir = "results"
subfolder_prefix = "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels"

print("Rebuilding Frankenstein filenames...")

fixed_count = 0
for filename in os.listdir(results_dir):
    if not filename.endswith(".pkl"): continue
    
    # 1. Find the true frame number using regex
    # It searches the mangled string for 'frame-0', 'frame-100', etc.
    match = re.findall(r'frame-(\d+)', filename)
    if not match: 
        match = re.findall(r'(\d+)\.pkl', filename)
        if not match: continue
        
    frame_str = match[-1] # Grabs the correct number
    
    # 2. Identify the true object type based on keywords hiding in the name
    name_lower = filename.lower()
    if "pcl" in name_lower: target_name = "lidar_pcl"
    elif "bev" in name_lower: target_name = "lidar_bev"
    elif "detections" in name_lower: target_name = "detections"
    elif "valid" in name_lower: target_name = "valid_labels"
    elif "performance" in name_lower: target_name = "det_performance"
    else: continue 
    
    # 3. Build the perfect Udacity filename
    new_name = f"{subfolder_prefix}__frame-{frame_str}__{target_name}.pkl"
    
    src = os.path.join(results_dir, filename)
    dst = os.path.join(results_dir, new_name)
    
    # 4. Rename it to the clean version
    if src != dst:
        os.rename(src, dst)
        fixed_count += 1
        
    # 5. Make the duplicate detections file the code requires
    if target_name == "detections":
        extra = f"{subfolder_prefix}__frame-{frame_str}__detections_resnet_0.5.pkl"
        extra_path = os.path.join(results_dir, extra)
        if not os.path.exists(extra_path):
            shutil.copy(dst, extra_path)

print(f"Success! Un-mangled and fixed {fixed_count} files. They are perfect now.")