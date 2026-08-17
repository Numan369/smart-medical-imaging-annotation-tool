export const PRODUCT_NAME = "Smart Medical Image Annotation Tool";
export const BRAND_NAME = "MediMask AI";
export const APP_VERSION = "v1.0.0-prototype";

export const RESEARCH_DISCLAIMER =
  "Research prototype. AI-generated suggestions require human review by a qualified expert and must not be treated as a medical diagnosis.";

export const AI_ASSISTANCE_METADATA = {
  name: "AI Pneumothorax Assistant",
  defaultThreshold: 0.35,
  inputResolution: "512 × 512 px",
  modality: "Chest Radiograph (CXR)",
  targetCondition: "Pneumothorax Segmentation",
};

export const COLOR_TOKENS = {
  navy: "#0F2744",
  teal: "#0E7490",
  cyanAi: "#22D3EE",
  greenAccepted: "#16A34A",
  amberReview: "#D97706",
  redRejected: "#DC2626",
  manualEditing: "#F59E0B",
};

export const STATUS_DESCRIPTIONS = {
  unannotated: "No saved annotation",
  "awaiting-review": "AI suggestion requires human review",
  finalized: "Annotation reviewed and saved",
} as const;

export const KEYBOARD_SHORTCUTS = [
  { key: "B", description: "Select Brush tool" },
  { key: "P", description: "Select Polygon tool" },
  { key: "E", description: "Select Eraser tool" },
  { key: "V", description: "Select Mode" },
  { key: "H / Space", description: "Pan tool" },
  { key: "Ctrl + Z", description: "Undo annotation stroke" },
  { key: "Ctrl + Shift + Z", description: "Redo stroke" },
  { key: "+ / -", description: "Zoom in / Zoom out" },
  { key: "0", description: "Fit image to screen" },
  { key: "O", description: "Toggle mask overlay visibility" },
  { key: "Ctrl + S", description: "Save annotation" },
];
