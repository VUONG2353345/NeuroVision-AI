import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        prob = torch.softmax(output, dim=1)[0, target_class].item()
        pred_class = torch.argmax(output, dim=1).item()

        output[0, target_class].backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy(), prob, pred_class


def get_disease_suggestion(x, y, w, h, img_width=224, img_height=224):
    cx = x + w / 2  
    cy = y + h / 2  
    # Đã đảo lại mã màu theo chuẩn RGB của Matplotlib (Red, Green, Blue)
    if cx < img_width * 0.25 or cx > img_width * 0.75 or cy < img_height * 0.25:
        return "Cortical lesion / Meningioma", (255, 0, 0), "Red"   # 255 ở đầu là Red
    elif img_width * 0.35 < cx < img_width * 0.65 and img_height * 0.4 < cy < img_height * 0.7:
        return "Thalamic / Deep Brain lesion", (0, 0, 255), "Blue"  # 255 ở cuối là Blue
    else:
        return "White matter lesion / Tumor", (0, 255, 0), "Green"  # 255 ở giữa là Green


def _is_valid_contour(c, cam_brain_only, orig_uint8, h_img, w_img,
                      is_dark, VALID_Y_MIN, VALID_Y_MAX, CAM_MIN, ASPECT_MAX):
    """
    Bộ lọc 4 tầng cho mỗi contour:
      1. Diện tích hợp lệ
      2. Tâm Y nằm trong vùng não thực sự
      3. Tỉ lệ cạnh không quá dài / bẹt
      4. CAM mean đủ cao (AI thực sự chú ý)
    """
    area = cv2.contourArea(c)

    # --- 1. Diện tích ---
    if is_dark:
        if not (40 < area < 3500):   # 8000 -> 3500: loại hốc mắt lớn (G_367)
            return False
    else:
        if not (40 < area < 12000):
            return False

    x, y, w, h = cv2.boundingRect(c)

    # --- 2. Tâm Y ---
    cy_norm = (y + h / 2) / h_img
    if not (VALID_Y_MIN < cy_norm < VALID_Y_MAX):
        return False

    # --- 3. Aspect ratio ---
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > ASPECT_MAX:          # loại box quá dài / bẹt (G_710 #2)
        return False

    # --- 4. CAM mean ---
    pad = 2
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad);  y2 = min(h_img, y + h + pad)
    cam_mean = np.mean(cam_brain_only[y1:y2, x1:x2])
    if cam_mean < CAM_MIN:           # loại vùng AI không thực sự chú ý
        return False

    return True


def analyze_brain_ai_driven(input_tensor, model, original_img):
    orig_uint8 = (original_img * 255).astype(np.uint8)
    h_img, w_img = orig_uint8.shape

    # ============================================================
    # GRAD-CAM – tự động tìm target_layer cho nhiều loại backbone
    # ============================================================
    target_layer = None

    # ResNet / Wide-ResNet / ResNeXt
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
        target_layer = model.backbone.layer4
    elif hasattr(model, 'layer4'):
        target_layer = model.layer4
    # EfficientNet / MobileNet (timm features)
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
        target_layer = model.backbone.features[-1]
    elif hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
        # DenseNet
        target_layer = model.features.denseblock4
    # ConvNeXt
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'stages'):
        target_layer = model.backbone.stages[-1]
    elif hasattr(model, 'stages'):
        target_layer = model.stages[-1]
    # VGG / generic features
    elif hasattr(model, 'features'):
        target_layer = model.features[-1]

    # Fallback: Conv2d cuối cùng
    if target_layer is None:
        last_conv = None
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        target_layer = last_conv

    if target_layer is None:
        raise ValueError("[GradCAM] Không tìm được target_layer cho model này!")

    grad_cam = GradCAM(model, target_layer)
    cam, prob_abnormal, pred_class = grad_cam.generate(input_tensor, target_class=1)
    cam_resized = cv2.resize(cam, (w_img, h_img))

    # ==========================================================
    # 1. LOẠI BỎ VÙNG CỔ & VIỀN ẢNH (erosion 30×30)
    # ==========================================================
    _, binary_brain = cv2.threshold(orig_uint8, 10, 255, cv2.THRESH_BINARY)
    contours_brain, _ = cv2.findContours(
        binary_brain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    brain_mask = np.zeros_like(orig_uint8)
    brain_core = np.zeros_like(orig_uint8)

    if contours_brain:
        largest_brain = max(contours_brain, key=cv2.contourArea)
        cv2.drawContours(brain_mask, [largest_brain], -1, 255, thickness=cv2.FILLED)
        brain_core = cv2.erode(
            brain_mask, np.ones((30, 30), np.uint8), iterations=1
        )

    cam_brain_only = cam_resized * (brain_core / 255.0)
    if cam_brain_only.max() > 0:
        cam_brain_only = cam_brain_only / cam_brain_only.max()

    # Heatmap
    cam_heatmap   = np.power(cam_brain_only, 2.0)
    heatmap_color = cv2.applyColorMap(
        (cam_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    orig_rgb = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
    heatmap_final = np.where(
        brain_core[:, :, None] > 0,
        cv2.addWeighted(heatmap_color, 0.55, orig_rgb, 0.45, 0),
        orig_rgb,
    )

    # ==========================================================
    # 2. VÙNG SĂN LÙNG (AI attention zone)
    # ==========================================================
    sz_mask = (cam_brain_only > 0.20).astype(np.uint8) * 255
    sz_mask = cv2.morphologyEx(sz_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    search_zone = np.zeros_like(orig_uint8)
    sz_cnts, _ = cv2.findContours(sz_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if sz_cnts:
        cv2.drawContours(search_zone, sz_cnts, -1, 255, thickness=cv2.FILLED)
    search_zone = cv2.bitwise_and(search_zone, brain_core)

    # ==========================================================
    # 3. PHÂN VÙNG THEO CƯỜNG ĐỘ SÁNG
    # ==========================================================
    dark_mask   = cv2.inRange(orig_uint8, 1,   50)
    bright_mask = cv2.inRange(orig_uint8, 145, 255)

    valid_dark   = cv2.bitwise_and(dark_mask,   search_zone)
    valid_bright = cv2.bitwise_and(bright_mask, search_zone)

    valid_dark   = cv2.morphologyEx(valid_dark,   cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    valid_dark   = cv2.morphologyEx(valid_dark,   cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    valid_bright = cv2.morphologyEx(valid_bright, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    valid_bright = cv2.morphologyEx(valid_bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts_dark,   _ = cv2.findContours(valid_dark,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_bright, _ = cv2.findContours(valid_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ==========================================================
    # 4. BỘ LỌC 4 TẦNG
    # ---------------------------------------------------------
    #  VALID_Y_MIN = 0.20  → loại mép đỉnh sọ
    #  VALID_Y_MAX = 0.65  → loại cổ / tiểu não (sagittal & coronal)
    #  ASPECT_MAX  = 2.5   → loại box dài/bẹt bất thường
    #  CAM_MIN     = 0.28  → chỉ giữ vùng AI chú ý cao
    # ==========================================================
    VALID_Y_MIN = 0.20
    VALID_Y_MAX = 0.65
    ASPECT_MAX  = 2.5
    CAM_MIN     = 0.28

    valid_contours = []

    for c in cnts_dark:
        if _is_valid_contour(c, cam_brain_only, orig_uint8, h_img, w_img,
                             is_dark=True,
                             VALID_Y_MIN=VALID_Y_MIN, VALID_Y_MAX=VALID_Y_MAX,
                             CAM_MIN=CAM_MIN, ASPECT_MAX=ASPECT_MAX):
            valid_contours.append(c)

    for c in cnts_bright:
        if _is_valid_contour(c, cam_brain_only, orig_uint8, h_img, w_img,
                             is_dark=False,
                             VALID_Y_MIN=VALID_Y_MIN, VALID_Y_MAX=VALID_Y_MAX,
                             CAM_MIN=CAM_MIN, ASPECT_MAX=ASPECT_MAX):
            valid_contours.append(c)

    # ---------- Fallback: dùng peak CAM nếu không qua lọc ----------
    if len(valid_contours) == 0:
        peak = (cam_brain_only > 0.60).astype(np.uint8) * 255
        peak = cv2.morphologyEx(peak, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cnts_peak, _ = cv2.findContours(peak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts_peak:
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                cy_norm = (y + h / 2) / h_img
                if VALID_Y_MIN < cy_norm < VALID_Y_MAX:
                    valid_contours.append(c)

        # Bỏ lọc Y nếu vẫn rỗng (tránh mất kết quả hoàn toàn)
        if len(valid_contours) == 0:
            valid_contours = [c for c in cnts_peak if cv2.contourArea(c) > 50]

    # ==========================================================
    # 5. VẼ BOUNDING BOX & TẠO BÁO CÁO
    # ==========================================================
    img_with_annotations = orig_rgb.copy()
    has_anomaly = pred_class == 1 and prob_abnormal > 0.5
    suggestions_list = []

    if has_anomaly:
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        box_count = 0

        for cnt in valid_contours[:5]:
            box_count += 1
            x, y, w, h = cv2.boundingRect(cnt)

            pad = 2
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad);  y2 = min(h_img, y + h + pad)

            disease_name, box_color, color_name = get_disease_suggestion(
                x1, y1, x2 - x1, y2 - y1, w_img, h_img
            )

            cv2.rectangle(img_with_annotations, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                img_with_annotations, f"#{box_count}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, box_color, 1, cv2.LINE_AA,
            )

            local_conf = min(
                np.mean(cam_brain_only[y1:y2, x1:x2]) * 100 + prob_abnormal * 10,
                99.9,
            )

            box_pixels = orig_uint8[y1:y2, x1:x2]
            box_mean   = np.mean(box_pixels) if len(box_pixels) > 0 else 100

            if box_mean < 60:
                subtype = "Necrotic/Cystic Hole"
            elif box_mean > 140:
                subtype = "Solid Enhancing Spot"
            else:
                subtype = "Tumor Core"

            coord_str    = f"X:{x1}➞{x2}, Y:{y1}➞{y2}"
            base_disease = disease_name.split("/")[0].strip()

            suggestions_list.append(
                f"• [#{box_count} - {color_name} Box] {subtype} ({base_disease}) "
                f"| AI Focus: {local_conf:.1f}% | Pos: [{coord_str}]"
            )

    return img_with_annotations, heatmap_final, prob_abnormal * 100, has_anomaly, suggestions_list