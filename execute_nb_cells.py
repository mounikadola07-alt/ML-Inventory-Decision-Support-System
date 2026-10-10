"""
execute_nb_cells.py - Lightweight notebook cell executor with image/png display capture
"""

import base64
import io
import json
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Executing notebook cells sequentially and capturing stream + display_data...")

with open("ML_Inventory_Decision_Support_System_improved.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

global_env = {}

code_cell_idx = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_cell_idx += 1
        src = "".join(cell.get("source", []))
        print(f"Executing Code Cell {code_cell_idx}...")
        
        plt.close('all')
        stdout_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_buf
        
        try:
            exec(src, global_env)
            sys.stdout = old_stdout
            out_text = stdout_buf.getvalue()
            
            outputs = []
            if out_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(keepends=True)
                })
            
            # Check for active matplotlib figures
            figs = [plt.figure(num) for num in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)
                print(f"  -> Captured matplotlib plot for Cell {code_cell_idx} ({len(img_b64)} b64 bytes)")
                
            cell["outputs"] = outputs
            cell["execution_count"] = code_cell_idx
            print(f"  -> Cell {code_cell_idx} SUCCESS ({len(out_text)} chars text, {len(figs)} figures)")
        except Exception as e:
            sys.stdout = old_stdout
            print(f"  -> Cell {code_cell_idx} FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

with open("ML_Inventory_Decision_Support_System_improved.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("\nAll code cells executed and outputs (including charts) saved successfully into ML_Inventory_Decision_Support_System_improved.ipynb!")
