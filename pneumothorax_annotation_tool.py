"""Desktop editor for reviewing and correcting AI pneumothorax masks."""

from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch

from infer_single_pneumothorax import (
    PREDICTION_THRESHOLD,
    choose_device,
    load_checkpoint,
    predict_mask,
    prepare_image,
)
from pneumothorax_model import PneumothoraxResNet34UNet


WINDOW_TITLE = "Smart Medical Imaging Annotation Tool"
DEFAULT_OUTPUT_DIRECTORY = Path("corrected_annotations")
MAX_UNDO_STATES = 20


class PneumothoraxAnnotationTool:
    """Provide a small human-in-the-loop editor for one DICOM at a time."""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.minsize(820, 700)

        self.device = choose_device()
        self.model = None
        self.dicom_path = None
        self.image = None
        self.model_input = None
        self.mask = None
        self.undo_states = []
        self.previous_point = None

        self.display_width = 0
        self.display_height = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.canvas_photo = None

        self.mode = tk.StringVar(value="draw")
        self.brush_size = tk.IntVar(value=24)
        self.show_overlay = tk.BooleanVar(value=True)
        self.status = tk.StringVar(
            value="Open a DICOM chest X-ray to begin."
        )

        self._build_interface()
        self._bind_shortcuts()

    def _build_interface(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")

        ttk.Button(
            toolbar,
            text="Open DICOM",
            command=self.open_dicom,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            toolbar,
            text="AI Suggest",
            command=self.generate_suggestion,
        ).pack(side="left", padx=6)

        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=8
        )

        ttk.Radiobutton(
            toolbar,
            text="Draw",
            value="draw",
            variable=self.mode,
        ).pack(side="left", padx=4)

        ttk.Radiobutton(
            toolbar,
            text="Erase",
            value="erase",
            variable=self.mode,
        ).pack(side="left", padx=4)

        ttk.Label(toolbar, text="Brush:").pack(side="left", padx=(10, 2))
        ttk.Scale(
            toolbar,
            from_=4,
            to=80,
            variable=self.brush_size,
            orient="horizontal",
            length=110,
        ).pack(side="left")

        ttk.Button(toolbar, text="Undo", command=self.undo).pack(
            side="left", padx=(10, 4)
        )
        ttk.Button(toolbar, text="Clear", command=self.clear_mask).pack(
            side="left", padx=4
        )

        ttk.Checkbutton(
            toolbar,
            text="Show mask",
            variable=self.show_overlay,
            command=self.render_image,
        ).pack(side="left", padx=8)

        ttk.Button(
            toolbar,
            text="Save Mask As...",
            command=self.save_mask,
        ).pack(side="right")

        heading = ttk.Label(
            self.root,
            text=(
                "AI-assisted pneumothorax annotation — "
                "human review required"
            ),
            font=("Segoe UI", 13, "bold"),
            anchor="center",
        )
        heading.pack(fill="x", padx=8, pady=(2, 6))

        canvas_frame = ttk.Frame(self.root, padding=(8, 0, 8, 0))
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#202020",
            highlightthickness=1,
            highlightbackground="#606060",
            cursor="crosshair",
        )
        self.canvas.pack(expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        ttk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            padding=8,
        ).pack(fill="x")

        ttk.Label(
            self.root,
            text=(
                "Yellow is the editable mask. The AI output is a suggestion, "
                "not a diagnosis or final annotation."
            ),
            anchor="center",
            foreground="#6a4b00",
            padding=(8, 0, 8, 8),
        ).pack(fill="x")

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda _event: self.open_dicom())
        self.root.bind("<Control-s>", lambda _event: self.save_mask())
        self.root.bind("<Control-z>", lambda _event: self.undo())

    def open_dicom(self):
        selected = filedialog.askopenfilename(
            title="Choose one chest X-ray DICOM file",
            filetypes=(("DICOM files", "*.dcm"), ("All files", "*.*")),
        )

        if not selected:
            return

        try:
            (
                image,
                model_input,
                height,
                width,
                converted_from_colour,
            ) = prepare_image(Path(selected))
        except Exception as error:
            messagebox.showerror("Could not open DICOM", str(error))
            return

        self.dicom_path = Path(selected)
        self.image = image
        self.model_input = model_input
        self.mask = np.zeros((height, width), dtype=bool)
        self.undo_states.clear()
        self._set_display_dimensions(width, height)
        self.render_image()
        colour_note = (
            " Converted from RGB to grayscale."
            if converted_from_colour
            else ""
        )
        self.status.set(
            f"Loaded {self.dicom_path.name} ({width} x {height})."
            f"{colour_note} Choose AI Suggest or draw manually."
        )

    def _set_display_dimensions(self, width, height):
        max_width = min(1000, max(600, self.root.winfo_screenwidth() - 180))
        max_height = min(760, max(480, self.root.winfo_screenheight() - 260))
        scale = min(max_width / width, max_height / height, 1.0)

        self.display_width = max(1, int(round(width * scale)))
        self.display_height = max(1, int(round(height * scale)))
        self.scale_x = width / self.display_width
        self.scale_y = height / self.display_height
        self.canvas.configure(
            width=self.display_width,
            height=self.display_height,
            scrollregion=(0, 0, self.display_width, self.display_height),
        )

    def _load_model_if_needed(self):
        if self.model is not None:
            return

        self.status.set(
            f"Loading validated model on {self.device}..."
        )
        self.root.update_idletasks()

        model = PneumothoraxResNet34UNet(
            use_pretrained_encoder=False,
            freeze_encoder=False,
        ).to(self.device)
        load_checkpoint(model, self.device)
        model.eval()
        self.model = model

    def generate_suggestion(self):
        if self.image is None:
            messagebox.showinfo(
                "Open a DICOM first",
                "Choose a DICOM chest X-ray before generating a suggestion.",
            )
            return

        try:
            self._load_model_if_needed()
            self.status.set(
                f"Generating suggestion at threshold "
                f"{PREDICTION_THRESHOLD:.2f}..."
            )
            self.root.update_idletasks()
            _, suggested_mask = predict_mask(
                self.model,
                self.model_input,
                self.device,
                output_size=self.image.shape,
            )
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))
            self.status.set("Prediction failed; the current mask was preserved.")
            return

        self._push_undo_state()
        self.mask = suggested_mask
        self.show_overlay.set(True)
        self.render_image()

        pixels = int(self.mask.sum())
        percentage = 100.0 * pixels / self.mask.size
        self.status.set(
            f"AI suggestion: {pixels:,} pixels ({percentage:.2f}% of image). "
            "Review it with Draw and Erase before saving."
        )

    def _push_undo_state(self):
        if self.mask is None:
            return

        self.undo_states.append(self.mask.copy())
        if len(self.undo_states) > MAX_UNDO_STATES:
            self.undo_states.pop(0)

    def undo(self):
        if not self.undo_states:
            self.status.set("Nothing to undo.")
            return

        self.mask = self.undo_states.pop()
        self.render_image()
        self.status.set("Previous mask restored.")

    def clear_mask(self):
        if self.mask is None:
            return

        self._push_undo_state()
        self.mask.fill(False)
        self.render_image()
        self.status.set("Mask cleared. Use Undo to restore it.")

    def _canvas_to_image_point(self, canvas_x, canvas_y):
        image_x = int(np.clip(canvas_x * self.scale_x, 0, self.mask.shape[1] - 1))
        image_y = int(np.clip(canvas_y * self.scale_y, 0, self.mask.shape[0] - 1))
        return image_x, image_y

    def start_stroke(self, event):
        if self.mask is None:
            return

        self._push_undo_state()
        self.previous_point = self._canvas_to_image_point(event.x, event.y)
        self._paint_between(self.previous_point, self.previous_point)

    def continue_stroke(self, event):
        if self.mask is None or self.previous_point is None:
            return

        current_point = self._canvas_to_image_point(event.x, event.y)
        self._paint_between(self.previous_point, current_point)
        self.previous_point = current_point

    def end_stroke(self, _event):
        if self.previous_point is not None:
            pixels = int(self.mask.sum())
            self.status.set(f"Editable mask now contains {pixels:,} pixels.")
        self.previous_point = None

    def _paint_between(self, start, end):
        mask_image = Image.fromarray(self.mask.astype(np.uint8), mode="L")
        drawer = ImageDraw.Draw(mask_image)
        average_scale = (self.scale_x + self.scale_y) / 2.0
        width = max(1, int(round(self.brush_size.get() * average_scale)))
        value = 1 if self.mode.get() == "draw" else 0

        drawer.line((start, end), fill=value, width=width)
        radius = width // 2
        for point in (start, end):
            drawer.ellipse(
                (
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ),
                fill=value,
            )

        self.mask = np.asarray(mask_image, dtype=np.uint8).astype(bool)
        self.render_image()

    def render_image(self):
        if self.image is None:
            self.canvas.delete("all")
            return

        grayscale = Image.fromarray(
            np.clip(self.image * 255.0, 0, 255).astype(np.uint8),
            mode="L",
        ).resize(
            (self.display_width, self.display_height),
            Image.Resampling.BILINEAR,
        )
        displayed = grayscale.convert("RGBA")

        if self.show_overlay.get() and self.mask is not None:
            resized_mask = Image.fromarray(
                self.mask.astype(np.uint8) * 255,
                mode="L",
            ).resize(
                (self.display_width, self.display_height),
                Image.Resampling.NEAREST,
            )
            mask_array = np.asarray(resized_mask) > 0
            overlay_array = np.zeros(
                (self.display_height, self.display_width, 4),
                dtype=np.uint8,
            )
            overlay_array[mask_array] = (255, 220, 0, 135)
            overlay = Image.fromarray(overlay_array, mode="RGBA")
            displayed = Image.alpha_composite(displayed, overlay)

        self.canvas_photo = ImageTk.PhotoImage(displayed)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.canvas_photo)

    def save_mask(self):
        if self.mask is None or self.dicom_path is None:
            messagebox.showinfo(
                "No annotation to save",
                "Open a DICOM and review its mask first.",
            )
            return

        DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(
            r"[^A-Za-z0-9._-]+", "_", self.dicom_path.stem
        ).strip("._") or "dicom"
        suggested_path = (
            DEFAULT_OUTPUT_DIRECTORY / f"{safe_stem}_corrected_mask.png"
        )

        selected = filedialog.asksaveasfilename(
            title="Save reviewed binary mask",
            initialdir=str(DEFAULT_OUTPUT_DIRECTORY.resolve()),
            initialfile=suggested_path.name,
            defaultextension=".png",
            filetypes=(("PNG mask", "*.png"),),
        )

        if not selected:
            return

        output_path = Path(selected)
        Image.fromarray(
            self.mask.astype(np.uint8) * 255,
            mode="L",
        ).save(output_path)

        pixels = int(self.mask.sum())
        self.status.set(
            f"Saved reviewed binary mask: {output_path} ({pixels:,} pixels)."
        )
        messagebox.showinfo(
            "Mask saved",
            f"The reviewed full-size binary mask was saved to:\n{output_path}",
        )

def main():
    torch.set_grad_enabled(False)
    root = tk.Tk()
    PneumothoraxAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()