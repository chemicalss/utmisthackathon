"""
Branchseed Viewer — offline desktop GUI (Tkinter, standard library only).

Displays daughter-branch predictions produced by your pipeline in the
required schema:
    {
      "case_id": "subject001",
      "parent": {"instance_id": "aorta"},
      "daughters": [
        {"instance_id": "branch_001", "parent_instance_id": "aorta",
         "ostium_xyz_mm": [x,y,z], "seed_xyz_mm": [x,y,z],
         "radius_mm": r, "direction_xyz": [dx,dy,dz]}
      ]
    }

No external packages required — only Python's standard library
(tkinter, json, os, base64). Runs fully offline.

------------------------------------------------------------------
GRAPHICS MAP — where every visual lives, and where to extend it
------------------------------------------------------------------
  draw_vein_diagram()   -> 2D schematic aorta + branch markers (clickable)
  draw_axial()          -> cross-section panel for the selected branch
                           - supports a REAL CT slice: if a daughter dict
                             has "axial_png_base64" (base64 PNG bytes, no
                             "data:image/..." prefix needed), it is drawn
                             instead of the schematic circle. Add that key
                             from your export script when ready.
  draw_custom_graphic() -> empty hook — add a new draw_*() function here
                           and call it from on_branch_selected() so it
                           refreshes whenever the selection changes.
------------------------------------------------------------------
"""

import json
import os
import base64
import tkinter as tk
from tkinter import ttk, filedialog

# ---------------------------------------------------------------------
# Design tokens (kept consistent with the earlier web version)
# ---------------------------------------------------------------------
BG        = "#14181D"
PANEL     = "#1B2027"
PANEL_2   = "#20262E"
BORDER    = "#2A313B"
TEXT      = "#E8E4DC"
MUTED     = "#8B93A0"
ACCENT    = "#C4573F"
ACCENT_DIM= "#7A362A"
MONO_FONT = ("Consolas", 10)
SANS_FONT = ("Segoe UI", 10)

# ---------------------------------------------------------------------
# Starts empty — nothing shows until you load real files or watch a folder.
# ---------------------------------------------------------------------
DEFAULT_CASES = {}


class BranchseedViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Branchseed Viewer")
        self.geometry("1180x720")
        self.configure(bg=BG)
        self.minsize(940, 600)

        self.cases = dict(DEFAULT_CASES)   # case_id -> case dict
        self.current_case_id = None
        self.current_branch_id = None
        self.toggles = {"mask": True, "ostium": True, "arrow": True}

        self.watch_dir = None
        self.watch_stamps = {}   # filename -> (mtime, size)
        self.watch_job = None

        self._build_ui()
        if self.cases:
            self._select_case(next(iter(self.cases)))
        else:
            self._show_empty_state()

    # -----------------------------------------------------------------
    # UI scaffold
    # -----------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=PANEL)
        style.configure("TButton", background=PANEL_2, foreground=TEXT, borderwidth=1)
        style.map("TButton", background=[("active", ACCENT_DIM)])
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=SANS_FONT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=SANS_FONT)
        style.configure("Header.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))

        # ---- Header / toolbar ----
        header = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        header.pack(side="top", fill="x")
        ttk.Label(header, text="Branchseed Viewer", style="Header.TLabel", background=PANEL).pack(
            side="left", padx=(16, 6), pady=10)
        ttk.Label(header, text="aortic branch origin verification", style="Sub.TLabel", background=PANEL).pack(
            side="left", pady=10)

        btn_frame = tk.Frame(header, bg=PANEL)
        btn_frame.pack(side="right", padx=12)
        ttk.Button(btn_frame, text="Load case JSON…", command=self.load_json_files).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Watch output folder…", command=self.choose_watch_folder).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.status_var, style="Muted.TLabel", background=PANEL).pack(
            side="right", padx=8)

        # ---- Body: three panes ----
        body = tk.Frame(self, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        # Left: case rail
        rail = tk.Frame(body, bg=PANEL, width=190, highlightbackground=BORDER, highlightthickness=1)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)
        ttk.Label(rail, text="Cases", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(12, 4))
        self.case_listbox = tk.Listbox(rail, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0,
                                        selectbackground=ACCENT_DIM, font=MONO_FONT, activestyle="none")
        self.case_listbox.pack(fill="both", expand=True, padx=6, pady=4)
        self.case_listbox.bind("<<ListboxSelect>>", self._on_case_listbox_select)

        # Center: vein diagram + axial view
        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        self.center_title = ttk.Label(center, text="", style="Muted.TLabel", background=BG)
        self.center_title.pack(anchor="w", pady=(0, 8))

        vein_panel = tk.Frame(center, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        vein_panel.pack(fill="x")
        vein_bar = tk.Frame(vein_panel, bg=PANEL)
        vein_bar.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(vein_bar, text="Aorta + branches (2D)", style="Muted.TLabel").pack(side="left")
        self.vein_canvas = tk.Canvas(vein_panel, bg="#0E1116", width=560, height=260,
                                      highlightthickness=0)
        self.vein_canvas.pack(padx=10, pady=10, fill="x")

        axial_panel = tk.Frame(center, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        axial_panel.pack(fill="both", expand=True, pady=(14, 0))
        toggle_bar = tk.Frame(axial_panel, bg=PANEL)
        toggle_bar.pack(fill="x", padx=10, pady=(10, 0))
        self.toggle_vars = {}
        for key, label in (("mask", "Aorta mask"), ("ostium", "Ostium marker"), ("arrow", "Direction arrow")):
            var = tk.BooleanVar(value=True)
            self.toggle_vars[key] = var
            cb = tk.Checkbutton(toggle_bar, text=label, variable=var, command=self._on_toggle_changed,
                                 bg=PANEL, fg=TEXT, selectcolor=PANEL_2, activebackground=PANEL,
                                 activeforeground=TEXT, font=SANS_FONT, bd=0, highlightthickness=0)
            cb.pack(side="left", padx=(0, 12))

        axial_wrap = tk.Frame(axial_panel, bg=PANEL)
        axial_wrap.pack(expand=True, fill="both", pady=10)
        self.axial_canvas = tk.Canvas(axial_wrap, bg="#0E1116", width=220, height=220, highlightthickness=0)
        self.axial_canvas.pack(anchor="center")

        # Right: detail panel
        detail = tk.Frame(body, bg=PANEL, width=300, highlightbackground=BORDER, highlightthickness=1)
        detail.pack(side="left", fill="y")
        detail.pack_propagate(False)

        ttk.Label(detail, text="Detected branches", style="Muted.TLabel").pack(anchor="w", padx=14, pady=(14, 4))
        self.branch_listbox = tk.Listbox(detail, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0,
                                          selectbackground=ACCENT_DIM, font=MONO_FONT, activestyle="none",
                                          height=8)
        self.branch_listbox.pack(fill="x", padx=10)
        self.branch_listbox.bind("<<ListboxSelect>>", self._on_branch_listbox_select)

        ttk.Label(detail, text="Selected branch", style="Muted.TLabel").pack(anchor="w", padx=14, pady=(16, 4))
        self.coords_text = tk.Text(detail, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0,
                                    font=MONO_FONT, wrap="word", height=12)
        self.coords_text.pack(fill="both", expand=True, padx=10)
        self.coords_text.configure(state="disabled")

        self.copy_btn = ttk.Button(detail, text="Copy JSON", command=self.copy_selected_json)
        self.copy_btn.pack(fill="x", padx=10, pady=10)

        # keep vein diagram markers -> instance_id for click detection
        self._vein_hit_targets = []  # list of (item_id, instance_id)
        self.vein_canvas.bind("<Button-1>", self._on_vein_click)

    # -----------------------------------------------------------------
    # Loading real data
    # -----------------------------------------------------------------
    def load_json_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])
        if not paths:
            return
        loaded, errors = 0, []
        for p in paths:
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if "case_id" not in data or "daughters" not in data:
                    raise ValueError("missing case_id or daughters[]")
                self.cases[data["case_id"]] = data
                loaded += 1
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        self._refresh_case_listbox()
        if errors:
            self.status_var.set(f"Loaded {loaded}, {len(errors)} failed — {errors[0]}")
        else:
            self.status_var.set(f"Loaded {loaded} case(s)")
        if loaded:
            last_id = data.get("case_id")
            if last_id in self.cases:
                self._select_case(last_id)

    def choose_watch_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        self.watch_dir = d
        self.watch_stamps = {}
        if self.watch_job:
            self.after_cancel(self.watch_job)
        self._poll_watch_folder()

    def _poll_watch_folder(self):
        if self.watch_dir and os.path.isdir(self.watch_dir):
            updated = 0
            try:
                for name in os.listdir(self.watch_dir):
                    if not name.lower().endswith(".json"):
                        continue
                    full = os.path.join(self.watch_dir, name)
                    try:
                        stat = os.stat(full)
                    except OSError:
                        continue
                    stamp = (stat.st_mtime, stat.st_size)
                    if self.watch_stamps.get(name) == stamp:
                        continue
                    try:
                        with open(full, "r") as f:
                            data = json.load(f)
                        if "case_id" not in data or "daughters" not in data:
                            continue
                        self.cases[data["case_id"]] = data
                        self.watch_stamps[name] = stamp
                        updated += 1
                    except Exception:
                        pass  # probably mid-write; retry next poll
            except OSError:
                pass
            if updated:
                self._refresh_case_listbox()
                if self.current_case_id in self.cases:
                    self._select_case(self.current_case_id)
            import time
            self.status_var.set(f"Watching · last checked {time.strftime('%H:%M:%S')}"
                                 + (f" · +{updated} updated" if updated else ""))
        self.watch_job = self.after(2000, self._poll_watch_folder)

    # -----------------------------------------------------------------
    # Case / branch selection
    # -----------------------------------------------------------------
    def _show_empty_state(self):
        self.current_case_id = None
        self.current_branch_id = None
        self.center_title.config(text="No case loaded yet")
        self.branch_listbox.delete(0, "end")
        self.vein_canvas.delete("all")
        self.vein_canvas.create_text(
            int(self.vein_canvas["width"]) // 2, int(self.vein_canvas["height"]) // 2,
            text="Load a case JSON or watch a folder to begin", fill=MUTED, font=SANS_FONT)
        self.draw_axial_empty()
        self.coords_text.configure(state="normal")
        self.coords_text.delete("1.0", "end")
        self.coords_text.insert("end", "No case loaded yet.")
        self.coords_text.configure(state="disabled")

    def _refresh_case_listbox(self):
        self.case_listbox.delete(0, "end")
        for cid, c in self.cases.items():
            self.case_listbox.insert("end", f"{cid}  ({len(c['daughters'])})")
        self._case_order = list(self.cases.keys())

    def _on_case_listbox_select(self, event):
        sel = self.case_listbox.curselection()
        if not sel:
            return
        cid = self._case_order[sel[0]]
        self._select_case(cid)

    def _select_case(self, case_id):
        if not hasattr(self, "_case_order"):
            self._refresh_case_listbox()
        if case_id not in self.cases:
            return
        self.current_case_id = case_id
        c = self.cases[case_id]
        self.center_title.config(text=f"{case_id} — aortic segment")
        if case_id in self._case_order:
            idx = self._case_order.index(case_id)
            self.case_listbox.selection_clear(0, "end")
            self.case_listbox.selection_set(idx)
        first_id = c["daughters"][0]["instance_id"] if c["daughters"] else None
        self._select_branch(first_id)

    def _refresh_branch_listbox(self, case):
        self.branch_listbox.delete(0, "end")
        if not case["daughters"]:
            self.branch_listbox.insert("end", "  (no eligible daughters)")
            self.branch_listbox.itemconfig(0, fg=MUTED)
            return
        for d in case["daughters"]:
            self.branch_listbox.insert("end", f"  {d['instance_id']}   r={d['radius_mm']}mm")

    def _on_branch_listbox_select(self, event):
        sel = self.branch_listbox.curselection()
        if not sel:
            return
        case = self.cases[self.current_case_id]
        if not case["daughters"]:
            return
        d = case["daughters"][sel[0]]
        self._select_branch(d["instance_id"])

    def _select_branch(self, instance_id):
        self.current_branch_id = instance_id
        case = self.cases[self.current_case_id]
        self._refresh_branch_listbox(case)
        # re-select in listbox to reflect state
        for i, d in enumerate(case["daughters"]):
            if d["instance_id"] == instance_id:
                self.branch_listbox.selection_clear(0, "end")
                self.branch_listbox.selection_set(i)
                break
        self.draw_vein_diagram(case)
        self._render_selected(case)

    def _on_toggle_changed(self):
        for k, v in self.toggle_vars.items():
            self.toggles[k] = v.get()
        case = self.cases[self.current_case_id]
        self._render_selected(case)

    # -----------------------------------------------------------------
    # GRAPHICS: 2D vein diagram (schematic, clickable)
    # -----------------------------------------------------------------
    def draw_vein_diagram(self, case):
        cv = self.vein_canvas
        cv.delete("all")
        self._vein_hit_targets = []
        w = int(cv["width"])
        h = int(cv["height"])
        trunk_x = w // 2

        cv.create_rectangle(trunk_x - 9, 10, trunk_x + 9, h - 10,
                             fill="#3A4048", outline="#4A515B")

        daughters = case["daughters"]
        if not daughters:
            cv.create_text(trunk_x, h // 2, text="no branches", fill=MUTED, font=MONO_FONT)
            return

        zs = [d["ostium_xyz_mm"][2] for d in daughters]
        z_min, z_max = min(zs) - 15, max(zs) + 15

        for d in daughters:
            t = (d["ostium_xyz_mm"][2] - z_min) / (z_max - z_min)
            y = (h - 20) - t * (h - 40) + 10
            side = 1 if d["ostium_xyz_mm"][0] >= 0 else -1
            x0 = trunk_x + side * 9
            x1 = x0 + side * 46
            is_sel = d["instance_id"] == self.current_branch_id
            color = ACCENT if is_sel else MUTED

            cv.create_line(x0, y, x1, y, fill=color, width=2.4 if is_sel else 1.6)
            marker = cv.create_oval(x0 - 5, y - 5, x0 + 5, y + 5, fill=color, outline="")
            cv.create_text(x1 + (10 if side > 0 else -10), y, text=d["instance_id"].replace("branch_", "b"),
                            fill=TEXT if is_sel else MUTED, font=("Consolas", 8),
                            anchor="w" if side > 0 else "e")
            self._vein_hit_targets.append((marker, d["instance_id"], x0, y))

    def _on_vein_click(self, event):
        best_id, best_dist = None, 999
        for marker, instance_id, x0, y0 in self._vein_hit_targets:
            dist = ((event.x - x0) ** 2 + (event.y - y0) ** 2) ** 0.5
            if dist <= 10 and dist < best_dist:
                best_dist = dist
                best_id = instance_id
        if best_id:
            self._select_branch(best_id)

    # -----------------------------------------------------------------
    # GRAPHICS: axial cross-section (schematic, with real-image hook)
    # -----------------------------------------------------------------
    def draw_axial(self, branch):
        cv = self.axial_canvas
        cv.delete("all")
        cv.create_rectangle(0, 0, 220, 220, fill="#0E1116", outline="")
        cx, cy, aorta_r = 110, 110, 46

        # GRAPHICS HOOK: real CT slice from your backend export.
        # If branch["axial_png_base64"] is set (base64 PNG bytes, no
        # "data:image/..." prefix), it's drawn here instead of the schematic.
        has_real_image = bool(branch.get("axial_png_base64"))
        if has_real_image:
            try:
                img = tk.PhotoImage(data=branch["axial_png_base64"], format="png")
                self._axial_img_ref = img  # keep a reference so it isn't garbage-collected
                cv.create_image(0, 0, image=img, anchor="nw")
            except tk.TclError:
                has_real_image = False  # fall back to schematic if decoding fails

        if self.toggles["mask"] and not has_real_image:
            cv.create_oval(cx - aorta_r, cy - aorta_r, cx + aorta_r, cy + aorta_r,
                            fill="#DDD7C8", outline="")

        import math
        angle = math.atan2(branch["direction_xyz"][1], branch["direction_xyz"][0])
        bx = cx + math.cos(angle) * (aorta_r + 6)
        by = cy + math.sin(angle) * (aorta_r + 6)
        r = max(4, branch["radius_mm"] * 3)

        if self.toggles["mask"] and not has_real_image:
            cv.create_oval(bx - r, by - r, bx + r, by + r, fill="#DDD7C8", outline="")

        if self.toggles["ostium"]:
            ox = cx + math.cos(angle) * aorta_r
            oy = cy + math.sin(angle) * aorta_r
            cv.create_oval(ox - 3.5, oy - 3.5, ox + 3.5, oy + 3.5, fill=ACCENT, outline="")

        if self.toggles["arrow"]:
            ax2 = bx + math.cos(angle) * 22
            ay2 = by + math.sin(angle) * 22
            cv.create_line(bx, by, ax2, ay2, fill=ACCENT, width=2, arrow="last")

    def draw_axial_empty(self):
        cv = self.axial_canvas
        cv.delete("all")
        cv.create_rectangle(0, 0, 220, 220, fill="#0E1116", outline="")
        cv.create_oval(110 - 46, 110 - 46, 110 + 46, 110 + 46, fill="#DDD7C8", outline="")
        cv.create_text(110, 170, text="no branches", fill=MUTED, font=("Consolas", 9))

    # -----------------------------------------------------------------
    # GRAPHICS: extend here — add new draw_*() functions and call them
    # from _render_selected() below so they refresh with the data.
    # -----------------------------------------------------------------
    def draw_custom_graphic(self, case, branch):
        pass

    # -----------------------------------------------------------------
    # Coordinate panel + selected-branch rendering
    # -----------------------------------------------------------------
    def _render_selected(self, case):
        branch = None
        for d in case["daughters"]:
            if d["instance_id"] == self.current_branch_id:
                branch = d
                break

        self.coords_text.configure(state="normal")
        self.coords_text.delete("1.0", "end")

        if branch is None:
            self.coords_text.insert("end", "No eligible daughter branches\ndetected in this case.")
            self.coords_text.configure(state="disabled")
            self.draw_axial_empty()
            return

        def fmt_mm(arr):
            return "[" + ",  ".join(f"{v:.1f} mm" for v in arr) + "]"

        def fmt_unit(arr):
            return "[" + ",  ".join(f"{v:.2f}" for v in arr) + "]"

        lines = [
            ("instance_id", branch["instance_id"]),
            ("ostium — exits the aorta", fmt_mm(branch["ostium_xyz_mm"])),
            ("seed — 5mm into branch", fmt_mm(branch["seed_xyz_mm"])),
            ("radius", f"{branch['radius_mm']:.1f} mm"),
            ("direction (unit vector)", fmt_unit(branch["direction_xyz"])),
        ]
        for label, value in lines:
            self.coords_text.insert("end", f"{label}\n", ("label",))
            self.coords_text.insert("end", f"  {value}\n\n", ("value",))
        self.coords_text.tag_configure("label", foreground=MUTED, font=("Segoe UI", 8))
        self.coords_text.tag_configure("value", foreground=TEXT, font=MONO_FONT)
        self.coords_text.configure(state="disabled")

        self.draw_axial(branch)
        self.draw_custom_graphic(case, branch)

    def copy_selected_json(self):
        case = self.cases[self.current_case_id]
        branch = None
        for d in case["daughters"]:
            if d["instance_id"] == self.current_branch_id:
                branch = d
                break
        if branch is None:
            return
        self.clipboard_clear()
        self.clipboard_append(json.dumps(branch, indent=2))
        self.copy_btn.config(text="Copied")
        self.after(1200, lambda: self.copy_btn.config(text="Copy JSON"))


if __name__ == "__main__":
    app = BranchseedViewer()
    app.mainloop()