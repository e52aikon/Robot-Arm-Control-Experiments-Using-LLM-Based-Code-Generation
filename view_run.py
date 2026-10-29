"""MuJoCoの3Dビューアで、場面や記録した試行を再生する。macOSでは mjpython で起動する。

  mjpython ~/llm_arm/view_run.py --scene S3                 S3の障害物を置いた場面を見る
  mjpython ~/llm_arm/view_run.py S3_scratch_numeric_t1      その試行の最後の計画を再生
  mjpython ~/llm_arm/view_run.py S3_scratch_numeric_t1 1    1回目の計画を再生

ビューアの操作：左ドラッグで回転、右ドラッグで平行移動、スクロールでズーム。
再生は繰り返す。ウィンドウを閉じると終わる。
"""
import argparse
import json
import math
import os
import time

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_HZ, SPEED, SETTLE, HOLD = 50, 1.0, 1.0, 1.0
OBSTACLE_RGBA, CONTACT_RGBA = (0.85, 0.15, 0.1, 1.0), (1.0, 0.9, 0.0, 1.0)


def load_record(tag):
    with open(os.path.join(HERE, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["tag"] == tag:
                return r
    raise SystemExit(f"{tag} は results.jsonl にありません")


def joint_path(waypoints):
    step, cmds, prev = SPEED / CONTROL_HZ, [], (0.0, 0.0)
    for wp in waypoints:
        n = max(1, math.ceil(max(abs(wp[0] - prev[0]), abs(wp[1] - prev[1])) / step))
        cmds += [(prev[0] + (wp[0] - prev[0]) * i / n, prev[1] + (wp[1] - prev[1]) * i / n)
                 for i in range(1, n + 1)]
        prev = tuple(wp)
    return [(0.0, 0.0)] * int(HOLD * CONTROL_HZ) + cmds + [prev] * int(SETTLE * CONTROL_HZ)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag", nargs="?")
    p.add_argument("attempt", nargs="?", type=int)
    p.add_argument("--scene")
    p.add_argument("--no-viewer", action="store_true", help="画面を出さずに1回だけ再生（確認用）")
    a = p.parse_args()

    scenarios = json.load(open(os.path.join(HERE, "scenarios.json"), encoding="utf-8"))
    waypoints, assumed = [], None
    if a.tag:
        rec = load_record(a.tag)
        att = rec["attempts"][(a.attempt or len(rec["attempts"])) - 1]
        if not att.get("waypoints"):
            raise SystemExit(f"試行{att['n']}は経由点がありません（{att['reason']}）")
        sid, waypoints, assumed = rec["scenario"], att["waypoints"], att.get("assumed")
        print(f"{a.tag} 試行{att['n']}：{'合格' if att['passed'] else '不合格 - ' + att['reason']}")
    else:
        sid = a.scene or "S3"
    sc = scenarios[sid]

    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "arm.xml"))
    d = mujoco.MjData(m)
    ob, og = m.body("obstacle").id, m.geom("obstacle").id
    m.body_pos[m.body("target_marker").id][:2] = sc["target"]
    o = sc["obstacles"][0]
    m.body_pos[ob][:2], m.geom_size[og][0] = (o["x"], o["y"]), o["r"]
    if assumed:
        m.body_pos[m.body("ghost").id][:2] = (assumed[0]["x"], assumed[0]["y"])
        m.geom_size[m.geom("ghost").id][0] = assumed[0]["r"]
    arm = {m.geom("link1").id, m.geom("link2").id}
    n_sub = round(1.0 / (CONTROL_HZ * m.opt.timestep))
    cmds = joint_path(waypoints) if waypoints else [(0.0, 0.0)] * CONTROL_HZ

    def run_once(sync=None, running=lambda: True):
        mujoco.mj_resetData(m, d)
        m.geom_rgba[og] = OBSTACLE_RGBA
        touched = False
        for cmd in cmds:
            if not running():
                return
            t0 = time.time()
            d.ctrl[:] = cmd
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
                for i in range(d.ncon):
                    c = d.contact[i]
                    if og in (c.geom1, c.geom2) and {c.geom1, c.geom2} & arm:
                        m.geom_rgba[og] = CONTACT_RGBA
                        touched = True
            if sync:
                sync()
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.time() - t0)))
        if not waypoints:
            return
        tip = d.site_xpos[m.site("tip").id][:2]
        err = math.hypot(tip[0] - sc["target"][0], tip[1] - sc["target"][1]) * 100
        print(f"最終誤差 {err:.1f} cm / 接触 {'あり' if touched else 'なし'}")

    if a.no_viewer:
        run_once()
        return
    from mujoco import viewer
    with viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (0.1, 0.25, 0.25)
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.3, -45, -90
        while v.is_running():
            run_once(v.sync, v.is_running)
            if not waypoints:          # 場面を見るだけのときは、閉じるまでそのまま
                while v.is_running():
                    v.sync()
                    time.sleep(0.05)
            time.sleep(1.5)


if __name__ == "__main__":
    main()
