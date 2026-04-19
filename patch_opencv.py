import re

file_path = r"misc/objdet_tools.py"

print("Patching Udacity's drawing tools for Windows 64-bit...")

with open(file_path, "r") as f:
    code = f.read()

# This uses Regex to find every instance of corners_int[X, Y] 
# and wraps it in int() so OpenCV stops crashing.
patched_code = re.sub(r'(corners_int\[\d+,\s*\d+\])', r'int(\1)', code)

if code != patched_code:
    with open(file_path, "w") as f:
        f.write(patched_code)
    print("Success! The drawing tools are now safe for OpenCV.")
else:
    print("No changes needed or already patched.")