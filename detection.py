import numpy as np
import cv2

def rotate_keep_canvas(bgr, deg):
    # Поворот картинки вокруг центра, но без “подгонки” под новый размер канвы.
    # Т.е. просто крутим в рамках исходного w×h. Для паспортов норм,
    # потому что у нас дальше есть свой нормалайзер раскладки.
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def best_by(dets, name):
    # Из пачки детекций берём лучшую по conf по имени класса.
    # Если вдруг два одинаковых — возьмём ту, что с бОльшей уверенностью.
    cand = None
    for d in dets:
        if d["name"] == name and (cand is None or d["conf"] > cand["conf"]):
            cand = d
    return cand

def minrect_from_poly_or_bbox(img, poly, xyxy):
    # Пытаемся вычислить минимальный прямоугольник (minAreaRect).
    # Сначала используем полигон маски (если есть и он адекватный),
    # иначе — считаем по bbox: берём ROI, ищем Canny-границы, строим minAreaRect по точкам.
    if poly is not None and len(poly) >= 3:
        cnt = poly.reshape(-1,1,2).astype(np.float32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        return rect, box
    # если маски нет — fallback к bbox. тут аккуратно следим, чтобы не вылезти за края
    x1,y1,x2,y2 = list(map(int, xyxy))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    roi = img[y1:y2, x1:x2]
    # чуть притушим шум и найдём ребра. пороги подогнаны примерно “на глаз”
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 40, 120, L2gradient=True)
    pts = cv2.findNonZero(edges)
    if pts is None or len(pts) < 8:
        # если не нашли ничего внятного — вернём просто bbox как прямоугольник без наклона
        rect = (((x1+x2)/2.0, (y1+y2)/2.0), (max(1,x2-x1), max(1,y2-y1)), 0.0)
        box = cv2.boxPoints(rect).astype(np.float32)
    else:
        # тут minAreaRect уже по реальным краям. не забываем сместить координаты на оффсет ROI
        rect0 = cv2.minAreaRect(pts)
        box0 = cv2.boxPoints(rect0).astype(np.float32)
        box0[:,0] += x1; box0[:,1] += y1
        (cx,cy),(w,h),ang = rect0
        rect = ((cx + x1, cy + y1), (w, h), ang)
        box = box0
    return rect, box

def angle_from_vertical(rect):
    # Преобразуем угол из minAreaRect в “насколько это вертикально”.
    # В OpenCV угол — к горизонтали, плюс ещё пляски когда w<h.
    # Здесь возвращаем угол относительно вертикали в диапазоне (-90; 90].
    (cx, cy), (w, h), ang = rect
    if w <= 0 or h <= 0: return 0.0
    angle_h = float(ang)
    if w < h:
        angle_h += 90.0
    angle_v = angle_h - 90.0
    while angle_v <= -90: angle_v += 180
    while angle_v > 90: angle_v -= 180
    return angle_v

def crop_from_xyxy(img, xyxy, expand_frac=0.10):
    # Простой кроп по bbox с небольшим расширением по X и совсем чуть-чуть по Y.
    # По паспорту так удобнее — поджимает поля, но не захватывает лишний мусор.
    x1,y1,x2,y2 = list(map(int, xyxy))
    h, w = img.shape[:2]
    ww = x2-x1; hh = y2-y1
    dx = int(ww*expand_frac); dy = int(hh*0.06)
    x1 = max(0, x1-dx); y1 = max(0, y1-dy)
    x2 = min(w-1, x2+dx); y2 = min(h-1, y2+dy)
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2].copy()
    
    
    
def yolo_detect(model, bgr, imgsz, conf, iou, device):
    # Обёртка над .predict() ультралитикса. Ничего хитрого: считаем,
    # пробегаемся по коробкам, собираем дружелюбный словарь.
    # Если есть маски — кладём coords полигона (для minAreaRect и прочих хитростей).
    res = model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    id2name = {int(i): n for i, n in res.names.items()}
    dets = []
    if res.boxes is not None and len(res.boxes):
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i].item())
            confd = float(res.boxes.conf[i].item())
            x1,y1,x2,y2 = map(float, res.boxes.xyxy[i].tolist())
            poly = None
            # у сегментации бывают маски. у детекции — нет. поэтому проверяем каждый раз.
            if res.masks is not None and res.masks.xy is not None and i < len(res.masks.xy):
                poly = res.masks.xy[i]
            dets.append({
                "name": id2name.get(cid, f"cls_{cid}"),
                "conf": confd,
                "cid": cid,
                "xyxy": [x1,y1,x2,y2],
                "poly": poly
            })
    return dets, res

