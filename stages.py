import numpy as np
import cv2
from utils import rotate_keep_canvas, best_by, minrect_from_poly_or_bbox, angle_from_vertical
from detection import yolo_detect

# --- Parameters ---
# Максимальный угол, на который пробуем “покрутить” картинку на первом проходе,
# когда детекция совсем ничего не нашла. Шаг 1° — потому что дешево, а помогает.
SWEEP_MAX_DEG = 30
SWEEP_STEP = 1

# Цвета для отрисовки классов. Ничего сакрального, просто чтобы на картинке
# одинаковые типы полей были одного цвета. Если что — меняется легко.
CLASS_COLORS = {
    "passport": (0, 180, 255),
    "page1": (0, 255, 255),
    "page2": (0, 255, 255),
    "photo": (255, 200, 0),
    "mrz": (255, 0, 255),
    "series_number": (0, 200, 255),
    "birth_date": (0, 255, 0),
    "birth_place": (0, 255, 0),
    "fam": (0, 255, 0),
    "name": (0, 255, 0),
    "sername": (0, 255, 0),
    "issue_date": (0, 255, 0),
    "issued_by": (0, 255, 0),
    "issued_code": (0, 255, 0),
    "sex": (255, 0, 0),
}

# Порядок полей для стабильной сортировки при визуализации и сборке JSON.
# Это исключительно для удобочитаемости результатов.
ALL_FIELD_ORDER = [
    "passport", "page1", "page2", "photo", "mrz", "series_number",
    "birth_date", "birth_place", "fam", "name", "sername",
    "issue_date", "issued_by", "issued_code", "sex"
]

# --- Stage 1 ---
def scene_score_basic(dets, W, H):
    # Простой скорер “насколько сцена похожа на паспорт вообще”.
    # Если есть MRZ, фото, серия — хорошо. Чем увереннее детектор, тем выше балл.
    sc = 0.0
    have = {n: best_by(dets, n) for n in ("passport", "page1", "page2", "photo", "mrz", "series_number")}
    # Веса подбирались эмпирически на нескольких пачках. Можно вынести в конфиг.
    for n, w in (("passport", 2.0), ("page1", 1.2), ("page2", 1.2), ("photo", 2.0), ("mrz", 2.5), ("series_number", 1.5)):
        if have[n] is not None:
            sc += w * (0.5 + 0.5 * min(1.0, have[n]["conf"]))
    # MRZ ожидаем внизу и вытянутую по горизонтали — это добавляет очков.
    if have["mrz"] is not None:
        x1, y1, x2, y2 = have["mrz"]["xyxy"]
        cy = (y1 + y2) / 2.0
        sc += 1.8 * (cy / H)  # чем ниже центр — тем лучше
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        sc += 1.0 * max(0.0, (w / h) - 1.0)  # шире высоты — плюс
    return sc

def choose_best_angle_by_sweep(model, bgr, imgsz, conf, iou, device):
    # Сначала пробуем в исходной ориентации. Если детекций нет — вращаем на ±1..±SWEEP_MAX_DEG.
    H, W = bgr.shape[:2]
    dets0, _ = yolo_detect(model, bgr, imgsz, conf, iou, device)
    sc0 = scene_score_basic(dets0, W, H) if len(dets0) else -1e9
    best = (0, sc0, dets0)

    if len(dets0) == 0:
        angles = []
        for d in range(1, SWEEP_MAX_DEG + 1, SWEEP_STEP):
            angles.append(+d); angles.append(-d)
        # Вращаем в обе стороны: иногда пол-градуса вправо спасают кадр.
        for deg in angles:
            img = rotate_keep_canvas(bgr, deg)
            dets, _ = yolo_detect(model, img, imgsz, conf, iou, device)
            sc = scene_score_basic(dets, W, H) if len(dets) else -1e9
            if sc > best[1]:
                best = (deg, sc, dets)
    return best

def normalize_without_mirror(canvas, dets):
    # Переворачиваем холст на 180°, если MRZ “вверх ногами”.
    # Тут не пытаемся зеркалить — фото и MRZ должны иметь правильную ориентацию.
    H, W = canvas.shape[:2]
    d_mrz = best_by(dets, "mrz")
    if d_mrz is not None:
        x1, y1, x2, y2 = d_mrz["xyxy"]
        cy = (y1 + y2) / 2.0
        if cy < H / 2:
            canvas = cv2.rotate(canvas, cv2.ROTATE_180)
            # нужно перекинуть и координаты детекций — иначе всё съедет.
            for d in dets:
                x1, y1, x2, y2 = d["xyxy"]
                d["xyxy"] = [W - x2, H - y2, W - x1, H - y1]
                if d["poly"] is not None:
                    p = d["poly"].copy()
                    p[:, 0] = W - p[:, 0]
                    p[:, 1] = H - p[:, 1]
                    d["poly"] = p
    return canvas, dets

# --- Stage 2 ---
def rotate_dets_k90(dets, W, H, k):
    # Поворачиваем боксы/маски на k*90° (в ту же сторону, что и картинка).
    # Важно синхронно крутить и bbox, и polygon, иначе визуализация и OCR поедут.
    k = k % 4
    if k == 0:
        return dets, W, H
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        poly = d["poly"]
        if k == 1:  # 90 CW
            nx1, ny1 = H - y2, x1
            nx2, ny2 = H - y1, x2
            nW, nH = H, W
            if poly is not None:
                p = poly.copy()
                px, py = p[:, 0].copy(), p[:, 1].copy()
                p[:, 0] = H - py
                p[:, 1] = px
                poly = p
        elif k == 2:  # 180
            nx1, ny1 = W - x2, H - y2
            nx2, ny2 = W - x1, H - y1
            nW, nH = W, H
            if poly is not None:
                p = poly.copy()
                p[:, 0] = W - p[:, 0]
                p[:, 1] = H - p[:, 1]
                poly = p
        else:  # 270 CW
            nx1, ny1 = y1, W - x2
            nx2, ny2 = y2, W - x1
            nW, nH = H, W
            if poly is not None:
                p = poly.copy()
                px, py = p[:, 0].copy(), p[:, 1].copy()
                p[:, 0] = py
                p[:, 1] = W - px
                poly = p
        out.append({
            "name": d["name"],
            "conf": d["conf"],
            "cid": d["cid"],
            "xyxy": [float(nx1), float(ny1), float(nx2), float(ny2)],
            "poly": poly
        })
    return out, nW, nH

def scene_score_passport_layout(dets, W, H):
    # Скоринг конкретно “паспортной” раскладки (две страницы, фото, MRZ внизу, серия справа вертикально…)
    score = 0.0
    d_mrz = best_by(dets, "mrz")
    d_ph = best_by(dets, "photo")
    d_p1 = best_by(dets, "page1")
    d_p2 = best_by(dets, "page2")
    d_ser = best_by(dets, "series_number")

    # Альбомная ориентация для разворота паспорта — почти всегда правда.
    score += 4.0 * (1.0 if W > H else 0.0)

    # MRZ — как в stage1: чем ниже и чем горизонтальнее, тем лучше.
    if d_mrz is not None:
        x1, y1, x2, y2 = d_mrz["xyxy"]
        cy = (y1 + y2) / 2.0
        score += 3.5 * (cy / H)
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        score += 1.5 * max(0.0, (w / h) - 1.0)

    # Фото ожидаем слева и ниже центра (в классическом развороте).
    if d_ph is not None:
        x1, y1, x2, y2 = d_ph["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        score += 2.0 * (1.0 - cx / W)  # левее — лучше
        score += 1.0 * (cy / H)        # ниже — лучше

    # page1 сверху page2 — типовая книжечная логика.
    if d_p1 is not None and d_p2 is not None:
        y1c = (d_p1["xyxy"][1] + d_p1["xyxy"][3]) / 2.0
        y2c = (d_p2["xyxy"][1] + d_p2["xyxy"][3]) / 2.0
        score += 2.0 * (1.0 if y1c < y2c else 0.0)

    # Серия-номер — обычно справа и вертикальная плашка.
    if d_ser is not None:
        x1, y1, x2, y2 = d_ser["xyxy"]
        cx = (x1 + x2) / 2.0
        score += 1.2 * (cx / W)  # правее — лучше
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        score += 1.0 * max(0.0, (h / w) - 1.0)  # выше ширины — ближе к вертикальной колонке

    return score

def choose_best_k90_layout(bgr, dets):
    # Пробуем все k∈{0,90,180,270}. Считаем скор каждой ориентации и берём лучшую.
    H, W = bgr.shape[:2]
    best = None
    for k in (0, 1, 2, 3):
        if k == 0:
            dets_k, Wk, Hk = dets, W, H
            img_k = bgr
        else:
            # крутим саму картинку
            if k == 1:
                img_k = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                Wk, Hk = H, W
            elif k == 2:
                img_k = cv2.rotate(bgr, cv2.ROTATE_180)
                Wk, Hk = W, H
            else:
                img_k = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                Wk, Hk = H, W
            # и синхронно крутим детекции
            dets_k, Wk2, Hk2 = rotate_dets_k90(dets, W, H, k)
            Wk, Hk = Wk2, Hk2

        sc = scene_score_passport_layout(dets_k, Wk, Hk)
        if (best is None) or (sc > best[0]):
            best = (sc, k, img_k, dets_k, Wk, Hk)
    return best

# --- Stage 3 ---
def scene_score_photo_mode(dets, W, H):
    # Когда один разворот не подошёл (нет page1), переключаемся в “режим фото”.
    # Здесь проверяем другие признаки: фото портретное, MRZ под фото, текстовые блоки горизонтальные и т.п.
    score = 0.0
    d_p1 = best_by(dets, "page1")
    if d_p1 is not None:
        # Если page1 всё-таки видна — этот режим нам не нужен.
        return -1e9

    d_ph = best_by(dets, "photo")
    d_mr = best_by(dets, "mrz")
    d_sn = best_by(dets, "series_number")

    score += 4.0 * (1.0 if W > H else 0.0)  # всё равно предпочитаем landscape

    if d_ph is not None:
        x1, y1, x2, y2 = d_ph["xyxy"]
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        score += 2.5 * (1.0 if h > w else 0.0)  # портретность фото
        score += 1.2 * (1.0 - cx / W)          # левее — лучше
        score += 1.2 * (cy / H)                # ниже — лучше

    if d_mr is not None:
        x1, y1, x2, y2 = d_mr["xyxy"]
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        cy = (y1 + y2) / 2.0
        score += 2.0 * max(0.0, (w / h) - 1.0)  # MRZ вытянута по X
        if d_ph is not None:
            _, py1, _, py2 = d_ph["xyxy"]
            pcy = (py1 + py2) / 2.0
            score += 1.5 * (1.0 if cy > pcy else 0.0)  # MRZ ниже фото — хорошо

    # Текстовые блоки обычно горизонтальные прямоугольники, а не “стойки”.
    TEXT_FIELDS = {
        "birth_date", "birth_place", "fam", "name", "sername",
        "issue_date", "issued_by", "issued_code", "sex"
    }
    for tname in TEXT_FIELDS:
        d = best_by(dets, tname)
        if d is None:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        score += 0.6 * (1.0 if w > h else 0.0)

    # Серия — вертикальная колонка справа чаще, чем нет.
    if d_sn is not None:
        x1, y1, x2, y2 = d_sn["xyxy"]
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        score += 0.8 * (1.0 if h > w else 0.0)  # вертикальность
        score += 0.8 * (cx / W)                 # правее — лучше

    return score

def choose_best_k90_photo_mode(bgr, dets):
    # Абсолютно та же идея, что и для разворота: пробуем 0/90/180/270 и выбираем максимум по “фото-режиму”.
    H, W = bgr.shape[:2]
    best = None
    for k in (0, 1, 2, 3):
        if k == 0:
            dets_k, Wk, Hk = dets, W, H
            img_k = bgr
        else:
            if k == 1:
                img_k = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                Wk, Hk = H, W
            elif k == 2:
                img_k = cv2.rotate(bgr, cv2.ROTATE_180)
                Wk, Hk = W, H
            else:
                img_k = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                Wk, Hk = H, W
            dets_k, Wk2, Hk2 = rotate_dets_k90(dets, W, H, k)
            Wk, Hk = Wk2, Hk2

        sc = scene_score_photo_mode(dets_k, Wk, Hk)
        if (best is None) or (sc > best[0]):
            best = (sc, k, img_k, dets_k, Wk, Hk)
    return best

# --- Visualization ---
def draw_oriented(img, dets, out_path):
    # Рисуем минимальные прямоугольники (minAreaRect) вокруг каждого объекта,
    # подписываем имя, уверенность и угол к вертикали — чисто для отладки и sanity-check.
    vis = img.copy()
    out_items = []
    order = {n: i for i, n in enumerate(ALL_FIELD_ORDER)}
    dets_sorted = sorted(dets, key=lambda d: order.get(d["name"], 999))
    for d in dets_sorted:
        name, conf = d["name"], d["conf"]
        rect, box = minrect_from_poly_or_bbox(img, d["poly"], d["xyxy"])
        if rect is None or box is None:
            # Если вычислить угол не получилось — всё равно вернём bbox,
            # чтобы не терять объект в JSON.
            out_items.append({
                "name": name, "conf": round(conf, 3),
                "angle_from_vertical_deg": None,
                "box_points": None,
                "bbox_xyxy": list(map(int, d["xyxy"]))
            })
            continue
        ang_v = float(angle_from_vertical(rect))
        color = CLASS_COLORS.get(name, (0, 255, 0))
        pts = box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
        label = f"{name} {conf:.2f} | {ang_v:+.1f}° v"
        p0 = tuple(pts[0, 0].tolist())
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tx, ty = p0[0], max(0, p0[1] - 8)
        cv2.rectangle(vis, (tx, ty - th - 6), (tx + tw + 8, ty + 2), color, -1, cv2.LINE_AA)
        cv2.putText(vis, label, (tx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        out_items.append({
            "name": name,
            "conf": round(conf, 3),
            "angle_from_vertical_deg": round(ang_v, 2),
            "box_points": [[float(p[0]), float(p[1])] for p in box.tolist()],
            "bbox_xyxy": list(map(int, d["xyxy"]))
        })
    cv2.imwrite(str(out_path), vis)
    return vis, out_items

def draw_oriented_and_collect(img, dets, out_path):
    # Просто обертка: рисуем и сразу возвращаем items в стабильном порядке.
    vis, items = draw_oriented(img, dets, out_path)
    order = {n: i for i, n in enumerate(ALL_FIELD_ORDER)}
    items_sorted = sorted(items, key=lambda it: order.get(it["name"], 999))
    return vis, items_sorted

