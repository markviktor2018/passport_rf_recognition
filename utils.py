import numpy as np
import cv2

def rotate_keep_canvas(bgr, deg):
    """
    Поворачивает картинку на заданный угол, 
    сохраняя размер канвы (без обрезки краёв).
    borderMode=REPLICATE — значит, фон заполняется 
    повторением соседних пикселей, а не чёрным цветом.
    """
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def best_by(dets, name):
    """
    Из списка детекций выбираем лучший объект по имени (class name),
    то есть тот, у которого максимальная уверенность (conf).
    Если таких несколько — берём самый уверенный.
    """
    cand = None
    for d in dets:
        if d["name"] == name and (cand is None or d["conf"] > cand["conf"]):
            cand = d
    return cand

def minrect_from_poly_or_bbox(img, poly, xyxy):
    """
    Получаем минимальный описывающий прямоугольник (minAreaRect)
    либо по полигону (poly), если он есть, либо по bounding box (xyxy).
    
    Используется, чтобы оценить ориентацию объекта (например, текстовой зоны).
    Если полигон отсутствует — достраиваем прямоугольник по краям bbox.
    """
    if poly is not None and len(poly) >= 3:
        cnt = poly.reshape(-1,1,2).astype(np.float32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        return rect, box

    # fallback: работаем по bbox
    x1,y1,x2,y2 = list(map(int, xyxy))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)

    # если кривой bbox (нет площади) — выходим
    if x2 <= x1 or y2 <= y1:
        return None, None

    # Вырезаем ROI и ищем края, чтобы оценить форму точнее
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 40, 120, L2gradient=True)
    pts = cv2.findNonZero(edges)

    if pts is None or len(pts) < 8:
        # Если ничего не нашли — вернём прямоугольник ровно по bbox
        rect = (((x1+x2)/2.0, (y1+y2)/2.0), (max(1,x2-x1), max(1,y2-y1)), 0.0)
        box = cv2.boxPoints(rect).astype(np.float32)
    else:
        # Иначе строим minAreaRect по найденным точкам
        rect0 = cv2.minAreaRect(pts)
        box0 = cv2.boxPoints(rect0).astype(np.float32)
        # смещаем координаты в глобальную систему
        box0[:,0] += x1; box0[:,1] += y1
        (cx,cy),(w,h),ang = rect0
        rect = ((cx + x1, cy + y1), (w, h), ang)
        box = box0

    return rect, box

def angle_from_vertical(rect):
    """
    Считает угол наклона прямоугольника относительно вертикали.
    Нужно для того, чтобы выравнивать текст (MRZ, номер паспорта и т.п.)
    в нормальное положение.
    """
    (cx, cy), (w, h), ang = rect
    if w <= 0 or h <= 0:
        return 0.0

    angle_h = float(ang)
    if w < h:
        # У OpenCV minAreaRect угол может «скакать» в зависимости от ориентации.
        # Корректируем — если ширина меньше высоты, добавляем 90°
        angle_h += 90.0

    # Переводим к «вертикальному» углу
    angle_v = angle_h - 90.0
    while angle_v <= -90: angle_v += 180
    while angle_v > 90: angle_v -= 180

    return angle_v

def crop_from_xyxy(img, xyxy, expand_frac=0.10):
    """
    Вырезает область изображения по bbox с небольшим расширением рамки.
    - expand_frac = 10% по ширине
    - по высоте добавляем фиксированно ~6% от bbox
    
    Это помогает не обрезать буквы/цифры по краям.
    """
    x1,y1,x2,y2 = list(map(int, xyxy))
    h, w = img.shape[:2]
    ww = x2-x1; hh = y2-y1
    dx = int(ww*expand_frac); dy = int(hh*0.06)

    x1 = max(0, x1-dx); y1 = max(0, y1-dy)
    x2 = min(w-1, x2+dx); y2 = min(h-1, y2+dy)

    if x2 <= x1 or y2 <= y1:
        return None  # если область пустая — возвращаем None

    return img[y1:y2, x1:x2].copy()

