"""
╔══════════════════════════════════════════════════════════════════╗
║   Paralytic Patient Face Tracker & Communication System          ║
║   Head Movement + Eye Tracking + Blink → Natural Sentences       ║
║   Algorithms: Kalman Filter + EAR + PnP Pose Estimation          ║
║                                                                  ║
║   How the patient communicates:                                  ║
║     Blink once        → YES / confirm                            ║
║     Blink twice       → NO / cancel                              ║
║     Look left/right   → navigate options                         ║
║     Head up/down      → better / worse                           ║
║     Head tilt         → unsure / repeat                          ║
║     Eyes closed 2s+   → fatigue / needs rest                     ║
║                                                                  ║
║   Bug fixes vs previous version:                                 ║
║     • Partial blink zone thresholds corrected (reachable now)    ║
║     • fatigue_alert no longer resets every frame                 ║
║     • bstate_ref always current (not one frame stale)            ║
╚══════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install mediapipe opencv-python numpy

Controls:
    Q / ESC  → Quit
    R        → Recalibrate head pose baseline
    S        → Save session log to CSV
    C        → Clear sentence history
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque

from config import Config, CUE_SENTENCES, EMOTION_MAP
from vision import get_head_pose, eye_aspect_ratio, iris_gaze, FACE_LANDMARK_IDS, LEFT_EAR_IDX, RIGHT_EAR_IDX, LEFT_IRIS, RIGHT_IRIS, LEFT_EYE, RIGHT_EYE
from models import KalmanFilter1D, MedianBuffer, BlinkDetector, DwellTracker, SentenceBuilder
from ui import draw_panel, draw_text, draw_bar, draw_gauge, draw_crosshair, wrap_text, COLORS
from tts import speak_text

#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    mp_face   = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    img_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    focal=img_w
    cam_mat=np.array([[focal,0,img_w/2],[0,focal,img_h/2],[0,0,1]],dtype=np.float64)
    dist_c =np.zeros((4,1))

    kf_yaw=KalmanFilter1D(Config.KALMAN_PROCESS_NOISE,Config.KALMAN_MEASURE_NOISE)
    kf_pit=KalmanFilter1D(Config.KALMAN_PROCESS_NOISE,Config.KALMAN_MEASURE_NOISE)
    kf_rol=KalmanFilter1D(Config.KALMAN_PROCESS_NOISE,Config.KALMAN_MEASURE_NOISE)
    kf_lh=KalmanFilter1D(1e-3,5e-2); kf_lv=KalmanFilter1D(1e-3,5e-2)
    kf_rh=KalmanFilter1D(1e-3,5e-2); kf_rv=KalmanFilter1D(1e-3,5e-2)

    ear_bl=MedianBuffer(Config.SMOOTH_WINDOW)
    ear_br=MedianBuffer(Config.SMOOTH_WINDOW)

    blink_det = BlinkDetector()
    dwell     = DwellTracker()
    sentence  = SentenceBuilder()

    baseline   = {"yaw":0.0,"pitch":0.0,"roll":0.0}
    calibrated = False

    yaw_s=pitch_s=roll_s=0.0
    lh_s=lv_s=rh_s=rv_s=0.0
    ear_l=ear_r=ear_avg=0.3
    gaze_h=gaze_v=0.0

    prev_time=time.time(); fps_buf=deque(maxlen=30); log_rows=[]

    latest_sentence=""; latest_short=""; latest_emotion=""; latest_ts=0.0

    WIN="Patient Communication Tracker"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN,1600,900)

    print("═"*65)
    print("  Paralytic Patient Communication Tracker")
    print("  R=Recalibrate  S=Save log  C=Clear history  Q/ESC=Quit")
    print("═"*65)

    result = None  # ensure defined before first draw

    while True:
        ret,frame=cap.read()
        if not ret: break

        frame=cv2.flip(frame,1)
        now=time.time(); now_ms=now*1000.0
        dt=now-prev_time; prev_time=now
        fps_buf.append(1.0/(dt+1e-6)); fps=float(np.mean(fps_buf))

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=face_mesh.process(rgb)

        active_cmd="neutral"; confirmed=False

        if result.multi_face_landmarks:
            lm=result.multi_face_landmarks[0].landmark

            # Head pose
            pose=get_head_pose(lm,img_w,img_h,cam_mat,dist_c)
            if pose:
                yr,pr,rr=pose
                yr-=baseline["yaw"]; pr-=baseline["pitch"]; rr-=baseline["roll"]
                if not calibrated:
                    kf_yaw.reset(yr); kf_pit.reset(pr); kf_rol.reset(rr)
                    calibrated=True
                yaw_s=kf_yaw.update(yr)
                pitch_s=kf_pit.update(pr)
                roll_s=kf_rol.update(rr)

            # EAR
            ear_l=ear_bl.update(eye_aspect_ratio(lm,LEFT_EAR_IDX, img_w,img_h))
            ear_r=ear_br.update(eye_aspect_ratio(lm,RIGHT_EAR_IDX,img_w,img_h))
            ear_avg=(ear_l+ear_r)/2.0
            bstate=blink_det.update(ear_avg,now_ms)   # FIX: bstate is always current

            # Iris
            try:
                lhr,lvr=iris_gaze(lm,LEFT_IRIS, LEFT_EYE, img_w,img_h)
                rhr,rvr=iris_gaze(lm,RIGHT_IRIS,RIGHT_EYE,img_w,img_h)
                lh_s=kf_lh.update(lhr); lv_s=kf_lv.update(lvr)
                rh_s=kf_rh.update(rhr); rv_s=kf_rv.update(rvr)
            except Exception: pass
            gaze_h=(lh_s+rh_s)/2.0; gaze_v=(lv_s+rv_s)/2.0

            # ── Command priority ──────────────────────────────
            if bstate.is_fatigue:
                active_cmd="fatigue"; confirmed=True
                dwell.update("fatigue",now)
            elif bstate.double_blink:
                active_cmd="double_blink"; confirmed=True
                dwell.update("double_blink",now)
            elif bstate.just_blinked:
                active_cmd="blink"; confirmed=True
                dwell.update("blink",now)
            
            # Compute relative deliberate intensity ratios for Head and Gaze
            yaw_ratio   = abs(yaw_s) / Config.HEAD_THRESHOLD_YAW
            pitch_ratio = abs(pitch_s) / Config.HEAD_THRESHOLD_PITCH
            roll_ratio  = abs(roll_s) / Config.HEAD_THRESHOLD_ROLL
            max_head_ratio = max(yaw_ratio, pitch_ratio, roll_ratio)
            
            gaze_h_ratio = abs(gaze_h) / Config.IRIS_THRESHOLD_H
            gaze_v_ratio = abs(gaze_v) / Config.IRIS_THRESHOLD_V
            max_gaze_ratio = max(gaze_h_ratio, gaze_v_ratio)

            # Head Movement takes priority over droopy eyes
            if max_head_ratio > 1.0:
                if max_head_ratio == yaw_ratio:
                    active_cmd = "head_right" if yaw_s > 0 else "head_left"
                elif max_head_ratio == pitch_ratio:
                    active_cmd = "head_down" if pitch_s > 0 else "head_up"
                else:
                    active_cmd = "tilt_left" if roll_s > 0 else "tilt_right"
                confirmed = dwell.update(active_cmd, now)
                
            # Gaze takes priority over droopy eyes
            elif max_gaze_ratio > 1.0:
                if max_gaze_ratio == gaze_h_ratio:
                    active_cmd = "eye_right" if gaze_h > 0 else "eye_left"
                else:
                    active_cmd = "eye_down" if gaze_v > 0 else "eye_up"
                confirmed = dwell.update(active_cmd, now)
            
            # Partial closes pushed to the end so they don't break dwell logic
            elif bstate.is_partial:
                active_cmd="partial_blink"; confirmed=False
                dwell.update("partial_blink",now)
            elif bstate.is_closed:
                active_cmd="blink"; confirmed=False
                dwell.update("blink",now)
            else:
                active_cmd="neutral"
                dwell.update("neutral",now)

            # ── Update sentence ───────────────────────────────
            if confirmed and active_cmd not in ("neutral","partial_blink"):
                res=sentence.confirm(active_cmd,now)
                if res:
                    latest_sentence=res
                    latest_short=CUE_SENTENCES[active_cmd][0]
                    latest_emotion=EMOTION_MAP.get(active_cmd,"")
                    latest_ts=now
                    
                    # Speak the newly confirmed patient sentence aloud!
                    speak_text(res)

                    print(f"  [{time.strftime('%H:%M:%S')}] "
                          f"{active_cmd:15s} → {res}")

            # Log
            bst=blink_det.state
            log_rows.append({
                "time_s":     f"{now:.3f}","fps":f"{fps:.1f}",
                "yaw":        f"{yaw_s:.2f}","pitch":f"{pitch_s:.2f}",
                "roll":       f"{roll_s:.2f}","gaze_h":f"{gaze_h:.3f}",
                "gaze_v":     f"{gaze_v:.3f}","ear_l":f"{ear_l:.3f}",
                "ear_r":      f"{ear_r:.3f}","is_partial":bst.is_partial,
                "is_fatigue": bst.is_fatigue,
                "fatigue_dur":f"{bst.fatigue_duration_s:.2f}",
                "blinks":     bst.blink_count,"command":active_cmd,
                "confirmed":  confirmed,"sentence":latest_sentence,
            })

        # ══════════════════════════════════════════════════════
        #  DRAW  (1600 × 900)
        # ══════════════════════════════════════════════════════
        display=np.zeros((900,1600,3),dtype=np.uint8)
        bst=blink_det.state   # FIX: read AFTER update, always current

        # Camera feed
        fh,fw=480,854
        display[40:40+fh,20:20+fw]=cv2.resize(frame,(fw,fh))
        cv2.rectangle(display,(20,40),(20+fw,40+fh),COLORS["accent"],2)

        # Fatigue overlay on camera  — FIX: uses bst.is_fatigue (persistent flag)
        if bst.is_fatigue:
            ov=display.copy()
            cv2.rectangle(ov,(20,40),(20+fw,40+fh),(30,30,200),-1)
            cv2.addWeighted(ov,0.42,display,0.58,0,display)
            draw_text(display,"FATIGUE / RESTING",
                      80,260,1.1,(30,50,255),True,2)
            draw_text(display,f"Eyes closed {bst.fatigue_duration_s:.1f}s",
                      180,320,0.72,COLORS["text"])
        elif bst.is_partial:
            # FIX: partial flag also now reachable
            draw_panel(display,22,42,340,30,0.9,COLORS["partial"])
            draw_text(display,"PARTIAL BLINK DETECTED",
                      30,62,0.55,COLORS["partial"],True)

        # Camera title
        cv2.rectangle(display,(20,10),(640,38),COLORS["panel"],-1)
        draw_text(display,"LIVE FEED  |  paralytic-optimised tracker",
                  28,29,0.48,COLORS["accent"])

        # Landmark dots
        if result and result.multi_face_landmarks:
            for idx in FACE_LANDMARK_IDS:
                px_=int(result.multi_face_landmarks[0].landmark[idx].x*img_w)
                py_=int(result.multi_face_landmarks[0].landmark[idx].y*img_h)
                cv2.circle(display,
                           (int(20+px_*fw/img_w),int(40+py_*fh/img_h)),
                           3,COLORS["accent"],-1,cv2.LINE_AA)
            for ig,ec in[(LEFT_IRIS,COLORS["blink"]),(RIGHT_IRIS,COLORS["warning"])]:
                for idx in ig:
                    px_=int(result.multi_face_landmarks[0].landmark[idx].x*img_w)
                    py_=int(result.multi_face_landmarks[0].landmark[idx].y*img_h)
                    cv2.circle(display,
                               (int(20+px_*fw/img_w),int(40+py_*fh/img_h)),
                               2,ec,-1,cv2.LINE_AA)

        # ── RIGHT PANEL — sensor data ─────────────────────────
        RX,RY,RW=894,10,340
        draw_panel(display,RX,RY,RW,520)
        draw_text(display,"SENSOR DATA",RX+10,RY+24,0.55,COLORS["accent"],True)
        draw_text(display,f"FPS {fps:.1f}",RX+250,RY+24,0.44,COLORS["dim"])

        draw_text(display,"HEAD ORIENTATION",RX+10,RY+52,0.43,COLORS["dim"])
        draw_gauge(display,RX+58, RY+105,36,yaw_s,  -45,45,"YAW",  COLORS["warning"])
        draw_gauge(display,RX+168,RY+105,36,pitch_s,-30,30,"PITCH",COLORS["green"])
        draw_gauge(display,RX+278,RY+105,36,roll_s, -30,30,"ROLL", COLORS["accent"])
        draw_bar(display,RX+10,RY+148,RW-20,9,yaw_s/45.0,  COLORS["warning"])
        draw_bar(display,RX+10,RY+162,RW-20,9,pitch_s/30.0,COLORS["green"])
        draw_bar(display,RX+10,RY+176,RW-20,9,roll_s/30.0, COLORS["accent"])

        draw_text(display,"EYE ASPECT RATIO",RX+10,RY+202,0.43,COLORS["dim"])
        for i,(lab,ev) in enumerate([("L",ear_l),("R",ear_r)]):
            bx=RX+10+i*160; bw=148
            pct=min(ev/0.4,1.0)
            if   ev < Config.EAR_BLINK_THRESHOLD: col=COLORS["blink"]
            elif ev < Config.EAR_PARTIAL_HIGH:     col=COLORS["partial"]
            else:                                   col=COLORS["accent"]
            cv2.rectangle(display,(bx,RY+210),(bx+bw,RY+224),COLORS["dim"],1)
            cv2.rectangle(display,(bx,RY+210),(bx+int(bw*pct),RY+224),col,-1)
            draw_text(display,f"{lab}:{ev:.3f}",bx+2,RY+238,0.40,col)

        # Blink indicator
        if bst.is_fatigue:  bi_col=COLORS["fatigue"]; bi_lab="FATIGUE"
        elif bst.is_partial:bi_col=COLORS["partial"];  bi_lab="PARTIAL"
        elif bst.is_closed: bi_col=COLORS["blink"];    bi_lab="CLOSED"
        else:               bi_col=COLORS["dim"];      bi_lab="OPEN"
        cv2.circle(display,(RX+20,RY+260),9,bi_col,-1,cv2.LINE_AA)
        draw_text(display,f"Blinks:{bst.blink_count}  [{bi_lab}]",
                  RX+36,RY+264,0.46,bi_col)

        # Fatigue timer bar
        if bst.is_closed and bst.fatigue_duration_s>0:
            frac=min(bst.fatigue_duration_s/(Config.BLINK_MAX_DURATION_MS/1000),1.0)
            fc=COLORS["fatigue"] if bst.is_fatigue else COLORS["warning"]
            cv2.rectangle(display,(RX+10,RY+274),(RX+RW-10,RY+282),COLORS["dim"],1)
            cv2.rectangle(display,(RX+10,RY+274),
                          (RX+10+int((RW-20)*frac),RY+282),fc,-1)
            draw_text(display,
                      f"Closed {bst.fatigue_duration_s:.1f}s / 2s limit",
                      RX+10,RY+296,0.36,fc)

        # Gaze boxes
        draw_text(display,"EYE GAZE",RX+10,RY+310,0.43,COLORS["dim"])
        for i,(lab,gh,gv) in enumerate([("L",lh_s,lv_s),("R",rh_s,rv_s)]):
            gx=RX+10+i*162; gy=RY+317; gw2=148; gh2=78
            draw_panel(display,gx,gy,gw2,gh2,0.85)
            draw_text(display,lab,gx+4,gy+13,0.37,COLORS["dim"])
            cx_=gx+gw2//2; cy_=gy+gh2//2
            draw_crosshair(display,cx_,cy_,7,COLORS["dim"])
            cv2.circle(display,
                       (int(cx_+gh*gw2*0.38),int(cy_+gv*gh2*0.38)),
                       5,COLORS["accent"],-1,cv2.LINE_AA)
        draw_bar(display,RX+10,RY+404,RW-20,9,gaze_h,COLORS["accent"])
        draw_bar(display,RX+10,RY+418,RW-20,9,gaze_v,COLORS["accent"])

        # Active command box
        if   active_cmd=="fatigue":                                   cmd_col=COLORS["fatigue"]
        elif active_cmd in("blink","double_blink","partial_blink"):   cmd_col=COLORS["blink"]
        elif active_cmd!="neutral":                                   cmd_col=COLORS["accent"]
        else:                                                         cmd_col=COLORS["dim"]
        draw_panel(display,RX+10,RY+434,RW-20,44,0.88,cmd_col)
        draw_text(display,CUE_SENTENCES.get(active_cmd,("?",""))[0],
                  RX+18,RY+462,0.68,cmd_col,True)
        prog=dwell.progress(now)
        cv2.rectangle(display,(RX+10,RY+480),(RX+RW-10,RY+486),COLORS["dim"],1)
        if prog>0:
            cv2.rectangle(display,(RX+10,RY+480),
                          (RX+10+int((RW-20)*prog),RY+486),cmd_col,-1)
        instant=active_cmd in("blink","double_blink","partial_blink","fatigue")
        draw_text(display,"Instant" if instant else f"Dwell {prog*100:.0f}%",
                  RX+10,RY+500,0.37,COLORS["dim"])
        if dwell.confirmed:
            draw_text(display,"CONFIRMED",RX+130,RY+500,0.46,COLORS["green"],True)

        # ── SENTENCE PANEL  (right, lower) ───────────────────
        SX,SY,SW,SH=894,540,690,350
        draw_panel(display,SX,SY,SW,SH,0.92,COLORS["sentence"])
        draw_text(display,"PATIENT IS SAYING",SX+10,SY+24,
                  0.55,COLORS["sentence"],True)
        if latest_ts>0:
            draw_text(display,f"{now-latest_ts:.0f}s ago",
                      SX+SW-90,SY+24,0.40,COLORS["dim"])

        draw_text(display,latest_short or "—",
                  SX+10,SY+70,1.05,COLORS["sentence"],True,2)

        if latest_sentence:
            for li,ln in enumerate(wrap_text(latest_sentence,52)[:3]):
                draw_text(display,ln,SX+10,SY+104+li*28,0.56,COLORS["text"])

        if latest_emotion:
            draw_text(display,f"State: {latest_emotion}",
                      SX+10,SY+196,0.43,COLORS["dim"])

        # Caregiver question
        cq=sentence.current_question()
        if cq:
            draw_panel(display,SX+10,SY+212,SW-20,65,0.95,COLORS["question"])
            draw_text(display,"CAREGIVER — ask the patient:",SX+16,SY+228,
                      0.40,COLORS["question"])
            draw_text(display,f'"{cq}"',SX+16,SY+252,0.55,COLORS["question"],True)
            draw_text(display,"Blink once = YES     Double blink = NO",
                      SX+16,SY+274,0.37,COLORS["dim"])

        # History
        draw_text(display,"RECENT",SX+10,SY+288,0.38,COLORS["dim"])
        for ri,(_,rcue,rsen) in enumerate(reversed(sentence.history[-5:])):
            a=max(55,175-ri*30); col=(a,a,a)
            sh=CUE_SENTENCES.get(rcue,("?",""))[0]
            draw_text(display,f"{sh}: {rsen[:55]}",
                      SX+10,SY+306+ri*20,0.36,col)

        # ── BOTTOM STRIP — large readable output for caregiver ─
        BY=530
        draw_panel(display,20,BY,854,360,0.78)
        draw_text(display,"WHAT THE PATIENT MEANS",30,BY+22,
                  0.55,COLORS["accent"],True)

        if latest_sentence:
            for li,ln in enumerate(wrap_text(latest_sentence,46)[:2]):
                draw_text(display,ln,30,BY+58+li*40,0.85,COLORS["sentence"],True,2)

        if latest_short:
            bw2=len(latest_short)*14+24
            cv2.rectangle(display,(30,BY+148),(30+bw2,BY+178),cmd_col,-1)
            draw_text(display,latest_short,38,BY+169,0.58,(0,0,0),True)

        if latest_emotion:
            draw_text(display,f"Detected state: {latest_emotion}",
                      30,BY+192,0.47,COLORS["dim"])

        if cq:
            draw_panel(display,30,BY+210,800,72,0.95,COLORS["question"])
            draw_text(display,"Ask the patient:",38,BY+228,0.42,COLORS["question"])
            draw_text(display,f'"{cq}"',38,BY+256,0.72,COLORS["question"],True,2)

        draw_text(display,
                  "R=Recalibrate  S=Save log  C=Clear history  Q/ESC=Quit",
                  30,BY+296,0.40,COLORS["dim"])
        draw_text(display,
                  f"YAW={yaw_s:+.1f}  PITCH={pitch_s:+.1f}  ROLL={roll_s:+.1f}"
                  f"  EAR={ear_avg:.3f}",
                  30,BY+316,0.38,COLORS["dim"])
        draw_text(display,
                  f"EAR zones: blink<{Config.EAR_BLINK_THRESHOLD}"
                  f"  partial<{Config.EAR_PARTIAL_HIGH}"
                  f"  open>{Config.EAR_OPEN_THRESHOLD}",
                  30,BY+336,0.35,COLORS["dim"])

        cv2.imshow(WIN,display)

        key=cv2.waitKey(1)&0xFF
        if key in(ord('q'),27): break
        elif key==ord('r'):
            if result and result.multi_face_landmarks:
                pose=get_head_pose(result.multi_face_landmarks[0].landmark,
                                   img_w,img_h,cam_mat,dist_c)
                if pose:
                    baseline["yaw"],baseline["pitch"],baseline["roll"]=pose
                    kf_yaw.reset(); kf_pit.reset(); kf_rol.reset()
                    print("  ✓ Recalibrated.")
        elif key==ord('s'):
            if log_rows:
                with open(Config.LOG_FILE,"w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=log_rows[0].keys())
                    w.writeheader(); w.writerows(log_rows)
                print(f"  ✓ Saved {len(log_rows)} rows → {Config.LOG_FILE}")
        elif key==ord('c'):
            sentence.clear()
            latest_sentence=latest_short=latest_emotion=""; latest_ts=0.0
            print("  ✓ History cleared.")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Session ended.")


if __name__=="__main__":
    main()
