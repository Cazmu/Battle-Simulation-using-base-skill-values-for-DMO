import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.widgets import Button

# ----------------------------- Helpers -----------------------------

def normalize_sp(sp_alloc, skills_data):
    # Remove disabled or invalid skills; keep name & valid priority fields
    cleaned = {"name": sp_alloc.get("name", "")}
    for sk in ("F1", "F2", "F3", "F4"):
        lvl = sp_alloc.get(sk, None)
        if sk in skills_data and isinstance(lvl, int) and lvl > 0:
            cleaned[sk] = lvl
    if sp_alloc.get("_use_priority") is not None:
        cleaned["_use_priority"] = sp_alloc["_use_priority"]
    if sp_alloc.get("_priority_order"):
        cleaned["_priority_order"] = [s for s in sp_alloc["_priority_order"] if s in skills_data]
    return cleaned


def get_sp_allocation(name, sp_cap, skills_data):
    # Prompt until user gives 4 ints under the SP cap; 0 disables a skill
    while True:
        try:
            raw = input(f"SP allocation for {name} (F1 F2 F3 F4): ").strip()
            parts = list(map(int, raw.split()))
            if len(parts) != 4:
                raise ValueError

            total_sp = 0
            for idx, lvl in enumerate(parts, start=1):
                sk = f"F{idx}"
                if sk in skills_data and lvl > 0:
                    cost = skills_data[sk].get('sp_cost', 1)
                    total_sp += cost * (lvl - 1)

            if total_sp > sp_cap:
                print(f"SP cap exceeded: used {total_sp} > {sp_cap}. Please re-enter.")
                continue
            return parts
        except ValueError:
            print("Invalid. Enter exactly four integers (e.g. `10 5 3 1`, or `0` to disable).")


# -------------------------- Core Simulation --------------------------

def simulate_battle(sp_allocation, skills_data, total_time):
    # Resolve per-skill damage from levels
    skill_levels = {}
    for skill, level in sp_allocation.items():
        if skill not in skills_data or skill == "name":
            continue
        base = skills_data[skill]['base_damage']
        inc  = skills_data[skill]['damage_per_level']
        dmg  = base + (level - 1) * inc
        skill_levels[skill] = {
            'damage':    dmg,
            'cooldown':  skills_data[skill]['cooldown'],
            'animation': skills_data[skill]['animation']
        }

    next_available_time = {sk: 0.0 for sk in skill_levels}
    usage_count        = {sk: 0   for sk in skill_levels}
    damage_by_skill    = {sk: 0   for sk in skill_levels}
    timeline = []

    current_time = 0.0
    total_damage = 0.0
    buff_stacks = 0
    buff_expire_time = -1.0

    use_priority   = sp_allocation.get("_use_priority", False)
    priority_order = sp_allocation.get("_priority_order", [])

    while current_time < total_time:
        # Skills that can start now and finish before total_time
        candidates = [
            sk for sk in skill_levels
            if next_available_time[sk] <= current_time
            and skill_levels[sk]['animation'] > 0
            and (current_time + skill_levels[sk]['animation'] <= total_time)
        ]

        if not candidates:
            # Jump to the next time any skill is available
            future_times = [t for sk, t in next_available_time.items() if t > current_time]
            if not future_times:
                break
            next_time = min(future_times)
            if next_time >= total_time:
                break
            current_time = next_time
            continue

        # Choose skill to use
        best_skill = None
        if sp_allocation.get("name") == "Bloom":
            if current_time >= buff_expire_time:
                buff_stacks = 0

            # Priority first
            if use_priority and priority_order:
                for sk in priority_order:
                    if sk not in candidates:
                        continue
                    if sk == "F2":
                        # Only use F2 under full stacks & enough remaining buff time
                        if buff_stacks == 3 and (buff_expire_time - current_time >= 8):
                            best_skill = "F2"
                            break
                        else:
                            continue
                    best_skill = sk
                    break

            # Fallback
            if best_skill is None:
                if buff_stacks < 3 and "F1" in candidates:
                    best_skill = "F1"
                else:
                    best_skill = max(candidates, key=lambda s: skill_levels[s]['damage'])
        else:
            # Others
            if use_priority and priority_order:
                for sk in priority_order:
                    if sk in candidates:
                        best_skill = sk
                        break
                if best_skill is None:
                    best_skill = max(candidates, key=lambda s: skill_levels[s]['damage'])
            else:
                best_skill = max(candidates, key=lambda s: skill_levels[s]['damage'])

        # Apply chosen skill
        dmg  = skill_levels[best_skill]['damage']
        anim = skill_levels[best_skill]['animation']
        cd   = skill_levels[best_skill]['cooldown']
        start_time = current_time
        end_time   = start_time + anim

        total_damage += dmg
        usage_count[best_skill] += 1
        damage_by_skill[best_skill] += dmg

        buff_time_remaining = max(buff_expire_time - start_time, 0.0)
        buff_pct = buff_stacks * 10

        timeline.append({
            'skill':  best_skill,
            'start':  round(start_time, 2),
            'end':    round(end_time,   2),
            'damage': int(dmg),
            'buff_stacks': buff_stacks,
            'buff_pct':   buff_pct,
            'buff_time_remaining': round(buff_time_remaining, 2)
        })

        # Bloom F1 can add stacks and refresh buff timer
        if sp_allocation.get("name") == "Bloom" and best_skill == "F1":
            if random.random() < 0.5:
                buff_stacks = min(buff_stacks + 1, 3)
            buff_expire_time = end_time + 20.0

        next_available_time[best_skill] = start_time + cd
        current_time = end_time

    actual_time_used = current_time
    dps = total_damage / actual_time_used if actual_time_used > 0 else 0.0

    return {
        "Time_Used": round(actual_time_used, 2),
        "Total_Damage": int(total_damage),
        "DPS": round(dps, 2),
        "Usage_Counts": usage_count,
        "Damage_Totals": damage_by_skill,
        "Timeline": timeline
    }


# ------------------------------ Main ------------------------------

def main():
    battle_time = 70.0  # Real-time playback length (seconds)
    sp_cap      = 76

    # Data
    digimon_database = {
        "Bloom": {
            'name': "Bloom",
            'F1': {'base_damage': 21895,  'damage_per_level': 738,  'cooldown': 3.0,  'animation': 4.11, 'sp_cost': 2},
            'F2': {'base_damage': 43738,  'damage_per_level': 1072, 'cooldown': 9.0,  'animation': 3.12, 'sp_cost': 3},
            'F3': {'base_damage': 241467, 'damage_per_level': 3440, 'cooldown': 51.0, 'animation': 5.04, 'sp_cost': 4},
            'F4': {'base_damage': 10396,  'damage_per_level': 2039, 'cooldown': 29.0, 'animation': 5.03, 'sp_cost': 4}
        },
        "IPMA": {
            'name': "IPMA",
            'F1': {'base_damage': 22555,  'damage_per_level': 753,  'cooldown': 4.5, 'animation': 3.30, 'sp_cost': 2},
            'F2': {'base_damage': 46686,  'damage_per_level': 1402, 'cooldown': 8.5, 'animation': 4.24, 'sp_cost': 3},
            'F3': {'base_damage': 255703, 'damage_per_level': 3405, 'cooldown': 51.0, 'animation': 4.80, 'sp_cost': 4},
            'F4': {'base_damage': 21064,  'damage_per_level': 2771, 'cooldown': 30.0, 'animation': 4.33, 'sp_cost': 3}
        },
        "Eos lvl6": {
            'name': "Eos lvl6",
            'F1': {'base_damage': 24854,  'damage_per_level': 504,  'cooldown': 5.0, 'animation': 4.11, 'sp_cost': 2},
            'F2': {'base_damage': 42666,  'damage_per_level': 1219, 'cooldown': 9.0, 'animation': 3.12, 'sp_cost': 3},
            'F3': {'base_damage': 73997,  'damage_per_level': 2010, 'cooldown': 30.0, 'animation': 4.04, 'sp_cost': 3},
            'F4': {'base_damage': 151524, 'damage_per_level': 3810, 'cooldown': 60.0, 'animation': 5.03, 'sp_cost': 4}
        },
        "CM": {
            'name': "CM",
            'F1': {'base_damage': 21157,  'damage_per_level': 738,  'cooldown': 4.0, 'animation': 3.03, 'sp_cost': 2},
            'F2': {'base_damage': 48324,  'damage_per_level': 1224, 'cooldown': 9.0, 'animation': 3.80, 'sp_cost': 3},
            'F3': {'base_damage': 238027, 'damage_per_level': 3599, 'cooldown': 55.0, 'animation': 5.67, 'sp_cost': 4},
            'F4': {'base_damage': 10057,  'damage_per_level': 738,  'cooldown': 30.0, 'animation': 6.92, 'sp_cost': 4}
        },
        "Miko": {
            'name': "Miko",
            'F1': {'base_damage': 24597,  'damage_per_level': 787,  'cooldown': 5.0, 'animation': 3.00, 'sp_cost': 2},
            'F2': {'base_damage': 17694,  'damage_per_level': 1873, 'cooldown': 12.0, 'animation': 5.20, 'sp_cost': 3},
            'F4': {'base_damage': 0,      'damage_per_level': 0,    'cooldown': 70.0, 'animation': 6.80, 'sp_cost': 0}
        },
        "OMM": {
            'name': "OMM",
            'F1': {'base_damage': 20121,  'damage_per_level': 731,  'cooldown': 4.0, 'animation': 3.47, 'sp_cost': 2},
            'F2': {'base_damage': 48879,  'damage_per_level': 1072, 'cooldown': 11.0, 'animation': 4.57, 'sp_cost': 3},
            'F3': {'base_damage': 231611, 'damage_per_level': 3440, 'cooldown': 65.0, 'animation': 5.58, 'sp_cost': 4}
        },
        "X7SM": {
            'name': "X7SM",
            'F1': {'base_damage': 21253,  'damage_per_level': 463,  'cooldown': 5.0, 'animation': 3.00, 'sp_cost': 2},
            'F2': {'base_damage': 51688,  'damage_per_level': 1375, 'cooldown': 10.0, 'animation': 4.00, 'sp_cost': 3},
            'F3': {'base_damage': 259810, 'damage_per_level': 3284, 'cooldown': 60.0, 'animation': 5.00, 'sp_cost': 4},
            'F4': {'base_damage': 61060,  'damage_per_level': 2218, 'cooldown': 55.0, 'animation': 4.00, 'sp_cost': 3}
        },
        "Zeed": {
            'name': "Zeed",
            'F1': {'base_damage': 18103,  'damage_per_level': 763,  'cooldown': 5.5, 'animation': 4.00, 'sp_cost': 2},
            'F2': {'base_damage': 55007,  'damage_per_level': 1175, 'cooldown': 9.0, 'animation': 5.00, 'sp_cost': 3},
            'F3': {'base_damage': 19574,  'damage_per_level': 2818, 'cooldown': 46.0, 'animation': 4.14, 'sp_cost': 3},
            'F4': {'base_damage': 259831, 'damage_per_level': 3584, 'cooldown': 55.0, 'animation': 6.33, 'sp_cost': 4}
        },
        "Alphamon Extreme": {
            'name': "Alphamon Extreme",
            'F1': {'base_damage': 26481,  'damage_per_level': 772,  'cooldown': 4.0, 'animation': 4.05, 'sp_cost': 2},
            'F2': {'base_damage': 44370,  'damage_per_level': 1207, 'cooldown': 8.0, 'animation': 3.26, 'sp_cost': 2},
            'F3': {'base_damage': 235712, 'damage_per_level': 3507, 'cooldown': 49.0, 'animation': 4.17, 'sp_cost': 4},
            'F4': {'base_damage': 24139,  'damage_per_level': 3061, 'cooldown': 29.0, 'animation': 5.06, 'sp_cost': 3}
        },
        "Kizuna Destiny": {
            'name': "Kizuna Destiny",
            'F1': {'base_damage': 17005,  'damage_per_level': 1184, 'cooldown': 10.0,'animation': 3.08, 'sp_cost': 2},
            'F2': {'base_damage': 27128,  'damage_per_level': 1756, 'cooldown': 3.5, 'animation': 3.10, 'sp_cost': 2},
            'F3': {'base_damage': 82040,  'damage_per_level': 3299, 'cooldown': 30.0,'animation': 4.26, 'sp_cost': 3},
            'F4': {'base_damage': 247515, 'damage_per_level': 3844, 'cooldown': 45.0,'animation': 6.12, 'sp_cost': 3}
        }
    }

    # Choose who to compare
    print("\nAvailable Digimon:")
    print(", ".join(digimon_database.keys()))
    while True:
        raw = input("\nWho do you want to compare? (comma-separated names or 'ALL'): ").strip()
        if not raw:
            print("  Please type some names or 'ALL'.")
            continue
        if raw.upper() == "ALL":
            selected_names = list(digimon_database.keys())
            break

        name_map = {k.lower(): k for k in digimon_database.keys()}
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        invalid = [p for p in parts if p not in name_map]
        if invalid:
            print(f"  Not found: {', '.join(invalid)}")
            print("  Use the exact names shown above (case-insensitive).")
            continue

        selected_names = [name_map[p] for p in parts]
        break

    # Collect inputs, simulate, store results
    all_results = {}
    for name in selected_names:
        skills = digimon_database[name]
        print(f"\n→ Setting up SP for {name}:")
        print("  (Enter four integers for F1 F2 F3 F4 in that order; use 0 to DISABLE a skill)")
        parts = get_sp_allocation(name, sp_cap, skills)
        sp_alloc = {'F1': parts[0], 'F2': parts[1], 'F3': parts[2], 'F4': parts[3], "name": name}

        # Priorities (filtered to existing skills)
        if name == "Miko":
            sp_alloc["_use_priority"] = True
            sp_alloc["_priority_order"] = [s for s in ['F4', 'F1', 'F2', 'F3'] if s in skills]
        elif name in ("Bloom", "IPMA", "Kizuna Destiny"):
            while True:
                pr = input(f"  Enter priority order for {name} (e.g. `F4 F1`), or `N` for none: ").strip().upper()
                if pr == "N":
                    sp_alloc["_use_priority"] = False
                    break
                parts = pr.split()
                existing = [s for s in parts if s in skills]
                if existing and len(existing) == len(set(existing)):
                    sp_alloc["_use_priority"] = True
                    remaining = [s for s in ("F1","F2","F3","F4") if s in skills and s not in existing]
                    sp_alloc["_priority_order"] = existing + remaining
                    break
                print("  Invalid. Use only skills this Digimon has.")
        else:
            sp_alloc["_use_priority"] = False

        sp_alloc = normalize_sp(sp_alloc, skills)
        all_results[name] = simulate_battle(sp_alloc, skills, total_time=battle_time)

    # Comparison table
    names = list(all_results.keys())
    print("\nComparison Results\n")
    header = "Digimon               | " + " | ".join(f"{n:<12}" for n in names)
    print(header)
    print("Time Used (s)        | " + " | ".join(f"{all_results[n]['Time_Used']:<12.2f}" for n in names))
    print("Total Damage         | " + " | ".join(f"{all_results[n]['Total_Damage']:,}".ljust(12) for n in names))
    print("Overall DPS          | " + " | ".join(f"{all_results[n]['DPS']:<12.2f}" for n in names))

    # Usage tables
    for name, res in all_results.items():
        print(f"\n{name} Skill Usage")
        print(f"{'Skill':<6} {'#Uses':<6} {'Total Damage'}")
        for sk in ['F1','F2','F3','F4']:
            uses  = res["Usage_Counts"].get(sk, 0)
            total = res["Damage_Totals"].get(sk, 0)
            print(f"{sk:<6} {uses:<6} {total:,}")
        print()

        print(f"Timeline for {name}:")
        print(f"{'Start':<7} {'End':<7} {'Skill':<6} {'Damage':<8} {'Stacks':<6} {'Buff%':<6} {'Rem.':<7}")
        for e in res["Timeline"]:
            print(
                f"{e['start']:<7.2f} {e['end']:<7.2f} "
                f"{e['skill']:<6} {e['damage']:<8,} "
                f"{e['buff_stacks']:<6} {e['buff_pct']:<6} {e['buff_time_remaining']:<7.2f}"
            )

    # ---------------- Simultaneous real-time playback with manual STOP ----------------
    # Precompute cumulative curves from timelines
    series = {}
    ymax = 1.0
    for name, res in all_results.items():
        tl = sorted(res["Timeline"], key=lambda e: e["end"])
        if not tl:
            series[name] = (np.array([0.0]), np.array([0.0]))
            continue
        ends = np.array([0.0] + [e["end"] for e in tl], dtype=float)
        cum  = np.array([0]   + list(np.cumsum([e["damage"] for e in tl])), dtype=float)
        series[name] = (ends, cum)
        ymax = max(ymax, float(cum[-1]))

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Live Damage (Press Q/Esc to stop)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Damage")
    ax.set_xlim(0, battle_time)
    ax.set_ylim(0, ymax * 1.05)
    ax.grid(alpha=0.25)

    # Stop button
    btn_ax = fig.add_axes([0.86, 0.02, 0.12, 0.06])
    stop_button = Button(btn_ax, "Stop", hovercolor="0.9")
    stop_requested = {"flag": False}

    def on_stop(event):
        stop_requested["flag"] = True

    stop_button.on_clicked(on_stop)

    # Keyboard shortcuts: Q or Esc to stop
    def on_key(event):
        if event.key in ("q", "escape"):
            stop_requested["flag"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Also stop if the figure is closed
    def on_close(event):
        stop_requested["flag"] = True

    fig.canvas.mpl_connect("close_event", on_close)

    lines = {}
    labels = {}
    xs_hist = {}
    ys_hist = {}

    # Stagger label vertical offsets (in pixels) to reduce overlap
    y_offsets = np.linspace(12, -12, num=len(names)) if names else []

    for i, name in enumerate(names):
        (line,) = ax.plot([], [], linewidth=2, label=name, zorder=3)
        lines[name] = line
        xs_hist[name] = []
        ys_hist[name] = []
        labels[name] = ax.text(
            0, 0, name,
            fontsize=10, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
            zorder=5, transform=ax.transData
        )
    ax.legend(loc="upper left", fontsize="medium")

    def set_label_pos(label, x_data, y_data, x_pad=0.6, y_pad_px=0):
        # Clamp inside axes and nudge vertically in display pixels
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x = min(max(x_data + x_pad, x0), x1)
        y = min(max(y_data, y0 + 0.01*(y1-y0)), y1 - 0.01*(y1-y0))
        label.set_position((x, y))
        label.set_transform(ax.transData + ScaledTranslation(0, y_pad_px / 72.0, fig.dpi_scale_trans))

    # Real-time loop matched to wall-clock time; can be interrupted
    fps = 60.0
    frame_dt = 1.0 / fps
    t0 = time.perf_counter()
    next_tick = t0

    while True:
        if stop_requested["flag"] or not plt.fignum_exists(fig.number):
            break

        now = time.perf_counter()
        elapsed = now - t0
        if elapsed >= battle_time:
            break

        y_top = ax.get_ylim()[1]
        changed_ylim = False

        for i, name in enumerate(names):
            times, cum = series[name]
            y = float(np.interp(elapsed, times, cum))
            xs_hist[name].append(elapsed)
            ys_hist[name].append(y)
            lines[name].set_data(xs_hist[name], ys_hist[name])
            set_label_pos(labels[name], elapsed, y, x_pad=0.6, y_pad_px=(y_offsets[i] if len(y_offsets) > 0 else 0))
            if y > y_top:
                y_top = y
                changed_ylim = True

        if changed_ylim:
            ax.set_ylim(0, y_top * 1.05)

        ax.figure.canvas.draw_idle()
        plt.pause(0.001)

        # keep 1:1 with real time
        next_tick += frame_dt
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()

    # Snap to final values if we reached battle_time
    if not stop_requested["flag"]:
        for i, name in enumerate(names):
            times, cum = series[name]
            xs_hist[name].append(battle_time)
            ys_hist[name].append(float(np.interp(battle_time, times, cum)))
            lines[name].set_data(xs_hist[name], ys_hist[name])
            set_label_pos(labels[name], battle_time, ys_hist[name][-1], x_pad=0.6, y_pad_px=(y_offsets[i] if len(y_offsets) > 0 else 0))

    plt.ioff()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
