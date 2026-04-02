import cv2

COLORS = {
    "bg":      ( 15, 15,  20),
    "panel":   ( 28, 32,  40),
    "accent":  (  0,200, 160),
    "warning": (  0,140, 255),
    "text":    (220,220, 230),
    "dim":     (100,105, 115),
    "blink":   (  0,220, 255),
    "green":   ( 40,220, 100),
    "partial": ( 60,160, 255),
    "fatigue": ( 30, 30, 220),   
    "sentence":(180,240, 255),
    "question":(140,255, 200),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def draw_panel(img, x, y, w, h, alpha=0.6, border=None):
    ov=img.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),COLORS["panel"],-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
    cv2.rectangle(img,(x,y),(x+w,y+h),border or COLORS["accent"],1)

def draw_text(img, text, x, y, scale=0.55, color=None, bold=False, thick=1):
    cv2.putText(img,text,(x,y),FONT_BOLD if bold else FONT,
                scale,color or COLORS["text"],thick,cv2.LINE_AA)

def draw_bar(img, x, y, w, h, value, color):
    mid = x + w // 2
    cv2.rectangle(img, (x, y), (x + w, y + h), COLORS["dim"], 1)
    
    # FIX: Clamp the value between -1.0 and 1.0 to prevent overflowing the bar length
    value = max(-1.0, min(1.0, value))
    
    pix = int(abs(value) * (w // 2))
    if value >= 0: 
        cv2.rectangle(img, (mid, y + 1), (mid + pix, y + h - 1), color, -1)
    else:        
        cv2.rectangle(img, (mid - pix, y + 1), (mid, y + h - 1), color, -1)
    cv2.line(img, (mid, y), (mid, y + h), COLORS["text"], 1)

def draw_gauge(img, cx, cy, r, value, lo, hi, label, color):
    a = max(0, min(180, 180 - int(180 * (value - lo) / (hi - lo + 1e-6))))
    cv2.ellipse(img, (cx,cy), (r,r), 0, 180, 360, COLORS["dim"], 2)
    cv2.ellipse(img, (cx,cy), (r,r), 0, 180, 360 - a, color, 3)
    draw_text(img, f"{value:+.1f}", cx-18, cy+12, 0.48, color, True)
    draw_text(img, label, cx - len(label)*4, cy+27, 0.36, COLORS["dim"])

def draw_crosshair(img, cx, cy, size=9, color=(0,200,160)):
    cv2.line(img,(cx-size,cy),(cx+size,cy),color,1,cv2.LINE_AA)
    cv2.line(img,(cx,cy-size),(cx,cy+size),color,1,cv2.LINE_AA)
    cv2.circle(img,(cx,cy),2,color,-1,cv2.LINE_AA)

def wrap_text(text, max_chars):
    words=text.split(); lines=[]; line=""
    for w in words:
        if len(line)+len(w)+1<=max_chars: line=(line+" "+w).strip()
        else:
            if line: lines.append(line)
            line=w
    if line: lines.append(line)
    return lines
