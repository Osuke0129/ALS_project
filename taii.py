import os
import datetime
import cv2
import csv
import mediapipe as mp
import numpy as np
import json
from shutil import move  # rmtreeは使わない（moveは未使用でもOK）
from deep_sort_realtime.deepsort_tracker import DeepSort

# === 設定読み込み ===
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TRAINING_DIR = config["training_dir"]
PLAY_DIR     = config["play_dir"]
RESULT_DIR   = config["result_dir"]
KANZYA_DIR   = config["kanzya_dir"]

os.makedirs(KANZYA_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
# Poseは1個を使い回す
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ▼▼▼ （最小変更方針）グローバルtrackerは使わず、record_and_save_dual内でカメラごとに作成
# tracker = DeepSort(max_age=5)

# === 閾値設定 ===
MOTION_THRESHOLD = 10.0
ANGLE_THRESHOLD = 20  # 許容範囲 ±20°

# === カメラ選択 ===
def select_dual_cameras():
    print("\n[カメラ選択]")
    cam_id_upper = int(input("正面カメラ（上半身）の番号を入力してください: "))
    cam_id_lower = int(input("背面カメラ（下半身）の番号を入力してください: "))
    return cam_id_upper, cam_id_lower

# === DTW関連 ===
def dtw_path_ndim(seq1: np.ndarray, seq2: np.ndarray):
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf); cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(seq1[i-1]-seq2[j-1])
            cost[i,j] = d + min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1])
    path=[]; i,j=n,m
    while i>0 and j>0:
        path.append((i-1,j-1))
        direction = np.argmin([cost[i-1,j],cost[i,j-1],cost[i-1,j-1]])
        if direction==0: i-=1
        elif direction==1: j-=1
        else: i-=1; j-=1
    return path[::-1]

def warp_sequence_to_template(template: np.ndarray, target: np.ndarray):
    path = dtw_path_ndim(template, target)
    return np.array([target[j] for i, j in path])

# === 角度計算 ===
def calc_angle(a,b,c):
    ba=a-b; bc=c-b
    denom = (np.linalg.norm(ba)*np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cos = np.dot(ba,bc)/denom
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def extract_upper_angles(lm):
    lm = np.array(lm).reshape(33,4)[:,:3]
    return [
        calc_angle(lm[11],lm[13],lm[15]),  # 左肘
        calc_angle(lm[12],lm[14],lm[16])   # 右肘
    ]

def extract_lower_angles(lm):
    lm = np.array(lm).reshape(33,4)[:,:3]
    return [
        calc_angle(lm[23],lm[25],lm[27]),  # 左膝
        calc_angle(lm[24],lm[26],lm[28])   # 右膝
    ]

# === 追加: Poseランドマーク→BBox＆信頼度（可視性平均） ===
def bbox_from_landmarks(result, img_w, img_h, vis_th=0.0):
    """
    MediaPipe PoseのランドマークからDeepSort入力用の [x1,y1,x2,y2], conf を返す。
    ランドマークなし→(None, None)
    """
    if result is None or result.pose_landmarks is None:
        return None, None
    xs, ys, confs = [], [], []
    for lm in result.pose_landmarks.landmark:
        vis = float(getattr(lm, "visibility", 0.0))
        if vis >= vis_th:
            xs.append(int(lm.x * img_w))
            ys.append(int(lm.y * img_h))
            confs.append(vis)
    if not xs or not ys:
        return None, None
    x1, x2 = max(0, min(xs)), min(img_w-1, max(xs))
    y1, y2 = max(0, min(ys)), min(img_h-1, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None, None
    conf = float(np.mean(confs)) if confs else 0.5
    return [x1, y1, x2, y2], conf

# === 動作量 ===
def motion_magnitude(csv_path:str)->float:
    try:
        with open(csv_path, encoding="utf-8") as fh:
            header_cols = len(fh.readline().split(','))
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=range(1, header_cols))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) < 2:
            return 0.0
        diff = np.diff(data, axis=0)
        return float(np.sum(np.linalg.norm(diff, axis=1)))
    except Exception:
        return 0.0

# === 見本データ平均化（DTW整列→平均し保存） ===
def compute_and_save_average_angles_dual():
    upper_seqs, lower_seqs = [], []

    for root, _, files in os.walk(TRAINING_DIR):
        if os.path.abspath(root).startswith(os.path.abspath(KANZYA_DIR)):
            continue
        for f in files:
            if not f.endswith('.csv'):
                continue
            if not (f.startswith('angles_upper') or f.startswith('angles_lower')):
                continue

            path = os.path.join(root, f)

            mag = motion_magnitude(path)
            if mag < MOTION_THRESHOLD:
                print(f"[WARN] 動作量が小さいCSV（移動しません）: {path} (mag={mag:.3f})")

            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    header_cols = len(fh.readline().split(','))
                data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, header_cols))
                if data.ndim == 1:
                    data = data.reshape(1, -1)
            except Exception as e:
                print(f"[WARN] CSV読み込み失敗のためスキップ: {path} ({e})")
                continue

            data = data[~np.isnan(data).any(axis=1)]
            if data.size == 0:
                print(f"[WARN] 有効データなし（NaNのみ）: {path}")
                continue

            if f.startswith('angles_upper'):
                upper_seqs.append(data)
            elif f.startswith('angles_lower'):
                lower_seqs.append(data)

    def build_ref_and_save(seqs, joints, npy_name, csv_name):
        if not seqs:
            print(f"[INFO] {npy_name}: 対象CSVが見つかりません")
            return
        template = seqs[0]
        warped = [warp_sequence_to_template(template, s) for s in seqs]
        min_len = min(len(w) for w in warped)
        aligned = np.stack([w[:min_len] for w in warped], axis=0)
        ref = np.nanmean(aligned, axis=0)

        np.save(os.path.join(TRAINING_DIR, npy_name), ref)

        frame_idx = np.arange(ref.shape[0], dtype=int)
        out_csv = os.path.join(TRAINING_DIR, csv_name)
        np.savetxt(
            out_csv,
            np.column_stack([frame_idx, ref]),
            delimiter=',',
            header='frame,' + ','.join(joints),
            comments='',
            fmt=['%d'] + ['%.6f']*ref.shape[1]
        )
        print(f"[INFO] {npy_name} / {csv_name} 保存完了。サンプル数: {len(seqs)}, 形状: {ref.shape}")

    build_ref_and_save(upper_seqs, ['left_elbow','right_elbow'],
                       'ref_angles_upper.npy', 'ref_angles_upper.csv')
    build_ref_and_save(lower_seqs, ['left_knee','right_knee'],
                       'ref_angles_lower.npy', 'ref_angles_lower.csv')

# === 2台同時録画（最初に映った1人をDeepSortで追従） ===
def record_and_save_dual(folder_path, video_upper_path, video_lower_path, cam_id_upper, cam_id_lower):
    upper_folder = os.path.join(folder_path,"upper")
    lower_folder = os.path.join(folder_path,"lower")
    os.makedirs(upper_folder, exist_ok=True)
    os.makedirs(lower_folder, exist_ok=True)

    cap_upper = cv2.VideoCapture(cam_id_upper)
    cap_lower = cv2.VideoCapture(cam_id_lower)

    if not cap_upper.isOpened() or not cap_lower.isOpened():
        print("[ERROR] カメラを開けません。")
        return

    w_u,h_u=int(cap_upper.get(3)),int(cap_upper.get(4))
    w_l,h_l=int(cap_lower.get(3)),int(cap_lower.get(4))
    fps=12.5
    out_upper=cv2.VideoWriter(video_upper_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w_u,h_u))
    out_lower=cv2.VideoWriter(video_lower_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w_l,h_l))

    print('o で録画開始 / Esc で終了')
    recording=False; frame_id=0
    writers_upper={}; writers_lower={}

    # ▼ 追加：カメラごとにDeepSortを用意
    tracker_u = DeepSort(max_age=5)
    tracker_l = DeepSort(max_age=5)

    # ▼ 追加：「最初に映った1人」をロックして追従
    primary_id_u = None  # 上半身
    primary_id_l = None  # 下半身
    LOST_PATIENCE = 30   # 見失い許容フレーム
    lost_u = 0
    lost_l = 0

    # 録画開始時にCSVを必ず作る
    def ensure_csv_opened():
        if 'upper' not in writers_upper:
            csv_path_u=os.path.join(upper_folder,'angles_upper.csv')
            fu=open(csv_path_u,'w',newline='', encoding='utf-8'); wu=csv.writer(fu)
            wu.writerow(['frame','left_elbow','right_elbow'])
            writers_upper['upper']=(fu,wu)
            print(f"[INFO] CSV作成: {csv_path_u}")
        if 'lower' not in writers_lower:
            csv_path_l=os.path.join(lower_folder,'angles_lower.csv')
            fl=open(csv_path_l,'w',newline='', encoding='utf-8'); wl=csv.writer(fl)
            wl.writerow(['frame','left_knee','right_knee'])
            writers_lower['lower']=(fl,wl)
            print(f"[INFO] CSV作成: {csv_path_l}")

    while cap_upper.isOpened() and cap_lower.isOpened():
        ret_u,frame_u=cap_upper.read()
        ret_l,frame_l=cap_lower.read()
        if not ret_u or not ret_l: break

        # 上半身処理（Pose）
        rgb_u=cv2.cvtColor(frame_u,cv2.COLOR_BGR2RGB)
        res_u=pose.process(rgb_u)
        if res_u.pose_landmarks:
            mp_draw.draw_landmarks(frame_u,res_u.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        # 下半身処理（Pose）
        rgb_l=cv2.cvtColor(frame_l,cv2.COLOR_BGR2RGB)
        res_l=pose.process(rgb_l)
        if res_l.pose_landmarks:
            mp_draw.draw_landmarks(frame_l,res_l.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        # ▼ 追加：Poseランドマークから擬似検出 → DeepSort更新
        dets_u = []
        bbox_u, conf_u = bbox_from_landmarks(res_u, w_u, h_u, vis_th=0.0)
        if bbox_u is not None:
            dets_u.append([bbox_u, conf_u, 'person'])
        tracks_u = tracker_u.update_tracks(dets_u, frame=frame_u)

        dets_l = []
        bbox_l, conf_l = bbox_from_landmarks(res_l, w_l, h_l, vis_th=0.0)
        if bbox_l is not None:
            dets_l.append([bbox_l, conf_l, 'person'])
        tracks_l = tracker_l.update_tracks(dets_l, frame=frame_l)

        # 主被写体のロック／維持
        disp_u = frame_u.copy()
        disp_l = frame_l.copy()

        confirmed_u = [t for t in tracks_u if t.is_confirmed()]
        if primary_id_u is None and confirmed_u:
            primary_id_u = confirmed_u[0].track_id
            lost_u = 0
            print(f"[INFO] UpperCam 主被写体IDをロック: {primary_id_u}")
        if primary_id_u is not None:
            visible_u = any(t.track_id==primary_id_u and t.is_confirmed() for t in confirmed_u)
            if visible_u:
                lost_u = 0
            else:
                lost_u += 1
                if lost_u > LOST_PATIENCE and confirmed_u:
                    primary_id_u = confirmed_u[0].track_id
                    lost_u = 0
                    print(f"[INFO] UpperCam 主被写体IDを再ロック: {primary_id_u}")

        confirmed_l = [t for t in tracks_l if t.is_confirmed()]
        if primary_id_l is None and confirmed_l:
            primary_id_l = confirmed_l[0].track_id
            lost_l = 0
            print(f"[INFO] LowerCam 主被写体IDをロック: {primary_id_l}")
        if primary_id_l is not None:
            visible_l = any(t.track_id==primary_id_l and t.is_confirmed() for t in confirmed_l)
            if visible_l:
                lost_l = 0
            else:
                lost_l += 1
                if lost_l > LOST_PATIENCE and confirmed_l:
                    primary_id_l = confirmed_l[0].track_id
                    lost_l = 0
                    print(f"[INFO] LowerCam 主被写体IDを再ロック: {primary_id_l}")

        # 可視化（主被写体を強調）
        for t in confirmed_u:
            x1,y1,x2,y2 = [int(v) for v in t.to_ltrb()]
            color = (0,255,0) if (primary_id_u is not None and t.track_id==primary_id_u) else (0,180,0)
            cv2.rectangle(disp_u,(x1,y1),(x2,y2),color,2)
            cv2.putText(disp_u,f'ID:{t.track_id}',(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        for t in confirmed_l:
            x1,y1,x2,y2 = [int(v) for v in t.to_ltrb()]
            color = (255,0,0) if (primary_id_l is not None and t.track_id==primary_id_l) else (160,0,0)
            cv2.rectangle(disp_l,(x1,y1),(x2,y2),color,2)
            cv2.putText(disp_l,f'ID:{t.track_id}',(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        # 入力処理（録画開始時にCSV作成）
        cv2.imshow('UpperCam',disp_u)
        cv2.imshow('LowerCam',disp_l)
        key=cv2.waitKey(1)&0xFF
        if key==ord('o') and not recording:
            recording=True
            ensure_csv_opened()
            print('録画開始')
        elif key==27:
            break

        if recording:
            # 動画は常に保存
            out_upper.write(disp_u)
            out_lower.write(disp_l)

            # 主被写体が見えているときだけ角度記録。見えなければNaN
            if res_u.pose_landmarks and primary_id_u is not None and any(t.track_id==primary_id_u for t in confirmed_u):
                angles_u=extract_upper_angles([[lm.x,lm.y,lm.z,lm.visibility] for lm in res_u.pose_landmarks.landmark])
            else:
                angles_u=[np.nan, np.nan]
            writers_upper['upper'][1].writerow([frame_id]+angles_u)

            if res_l.pose_landmarks and primary_id_l is not None and any(t.track_id==primary_id_l for t in confirmed_l):
                angles_l=extract_lower_angles([[lm.x,lm.y,lm.z,lm.visibility] for lm in res_l.pose_landmarks.landmark])
            else:
                angles_l=[np.nan, np.nan]
            writers_lower['lower'][1].writerow([frame_id]+angles_l)

            frame_id+=1
            cv2.putText(disp_u,'● REC',(10,h_u-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(disp_l,'● REC',(10,h_l-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cap_upper.release(); cap_lower.release()
    out_upper.release(); out_lower.release()
    cv2.destroyAllWindows()
    for f,_ in writers_upper.values(): f.close()
    for f,_ in writers_lower.values(): f.close()

# === 比較 ===
def compare_after_record_dual(video_path, csv_path, ref, save_path_analysis, save_path_display, csv_out, joints):
    with open(csv_path, encoding="utf-8") as fh:
        header_cols = len(fh.readline().split(','))
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=range(1, header_cols))  # 角度列のみ
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # NaN行は除外（整列/統計が壊れないように）
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) == 0:
        print(f"[WARN] 有効データがありません（NaNのみ）: {csv_path}")
        return

    warped = warp_sequence_to_template(ref, data)

    cap=cv2.VideoCapture(video_path)
    w,h=int(cap.get(3)),int(cap.get(4))
    fps=12.5
    out_analysis=cv2.VideoWriter(save_path_analysis,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    out_display=cv2.VideoWriter(save_path_display,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))

    diffs=[]; all_diffs=[]; i=0
    while cap.isOpened() and i<len(warped):
        ret,frame=cap.read()
        if not ret: break
        frame_diffs=[]
        ref_idx = min(i, len(ref)-1)
        for j,(r,a) in enumerate(zip(ref[ref_idx],warped[i])):
            diff=a-r
            frame_diffs.append(diff)
            color=(0,255,0) if abs(diff)<=ANGLE_THRESHOLD else (0,0,255)
            cv2.putText(frame,f'{joints[j]}: {diff:+.1f}°',(50,50+j*40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        diffs.append(frame_diffs)
        all_diffs.append([i]+frame_diffs)
        out_analysis.write(frame)
        out_display.write(cv2.flip(frame,1))
        i+=1

    cap.release(); out_analysis.release(); out_display.release()
    diffs=np.array(diffs)
    rates=np.mean((np.abs(diffs)<=ANGLE_THRESHOLD).astype(int),axis=0)
    avg=np.mean(diffs,axis=0)

    with open(csv_out,'w',newline='', encoding='utf-8') as rf:
        w=csv.writer(rf)
        w.writerow(['joint','match_rate','avg_diff'])
        for j,(r,m) in zip(joints,zip(rates,avg)):
            w.writerow([j,f'{r:.2f}',f'{m:+.2f}'])
        w.writerow([])
        w.writerow(['frame']+[f'{j}_diff' for j in joints])
        for row in all_diffs:
            w.writerow(row)

# === 横並び動画 ===
def concat_videos_side_by_side(video1_path, video2_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 注意: 元コードのままならCAP_PROP_FRAME_HEIGHT, typo修正はしない
    fps = cap1.get(cv2.CAP_PROP_FPS) or 12.5

    h = min(h1, h2)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1+w2, h))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame1 = cv2.resize(frame1, (w1, h))
        frame2 = cv2.resize(frame2, (w2, h))
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

    cap1.release(); cap2.release(); out.release()
    print(f"[INFO] 横連結動画を保存: {output_path}")

# === MAIN ===
def main():
    print('[1] 見本作成  [2] 比較')
    mode = input('モード: ')
    user = input('名前: ')
    cam_id_upper, cam_id_lower = select_dual_cameras()

    while True:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(TRAINING_DIR if mode=='1' else PLAY_DIR, f'{user}_{ts}')
        os.makedirs(folder, exist_ok=True)

        try:
            video_upper = os.path.join(folder,"upper",f'video_upper_{user}_{ts}.mp4')
            video_lower = os.path.join(folder,"lower",f'video_lower_{user}_{ts}.mp4')
            record_and_save_dual(folder,video_upper,video_lower,cam_id_upper,cam_id_lower)

            if mode=='1':
                compute_and_save_average_angles_dual()

            elif mode=='2':
                ref_upper=np.load(os.path.join(TRAINING_DIR,'ref_angles_upper.npy'))
                ref_lower=np.load(os.path.join(TRAINING_DIR,'ref_angles_lower.npy'))
                result_sub=os.path.join(RESULT_DIR,f'{user}_{ts}')
                result_upper=os.path.join(result_sub,"upper")
                result_lower=os.path.join(result_sub,"lower")
                os.makedirs(result_upper,exist_ok=True)
                os.makedirs(result_lower,exist_ok=True)

                for root,_,files in os.walk(folder):
                    for f in files:
                        if f.startswith('angles_upper') and f.endswith('.csv'):
                            csv_path=os.path.join(root,f)
                            vis_a=os.path.join(result_upper,f'analysis_{f.replace(".csv",".mp4")}')
                            vis_d=os.path.join(result_upper,f'display_{f.replace(".csv",".mp4")}')
                            res_c=os.path.join(result_upper,f'result_{f}')
                            compare_after_record_dual(video_upper,csv_path,ref_upper,vis_a,vis_d,res_c,
                                                      ['left_elbow','right_elbow'])
                        elif f.startswith('angles_lower') and f.endswith('.csv'):
                            csv_path=os.path.join(root,f)
                            vis_a=os.path.join(result_lower,f'analysis_{f.replace(".csv",".mp4")}')
                            vis_d=os.path.join(result_lower,f'display_{f.replace(".csv",".mp4")}')
                            res_c=os.path.join(result_lower,f'result_{f}')
                            compare_after_record_dual(video_lower,csv_path,ref_lower,vis_a,vis_d,res_c,
                                                      ['left_knee','right_knee'])

                # 必要なら横並び動画を有効化
                # concat_videos_side_by_side(ref_video_upper, video_upper, os.path.join(result_upper, "side_by_side_upper.mp4"))
                # concat_videos_side_by_side(ref_video_lower, video_lower, os.path.join(result_lower, "side_by_side_lower.mp4"))

        except Exception as e:
            print(f"[ERROR] 撮影中にエラー発生: {e}")
            continue

        again = input("もう一度撮影しますか？ (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__=='__main__':
    main()
