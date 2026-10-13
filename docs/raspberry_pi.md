# Running on a Raspberry Pi

A Raspberry Pi 4 is the long term home for the runs. It draws little power, never sleeps, and runs each run as its own systemd service that restarts after a crash or a power cut and resumes from its last checkpoint. This guide covers the hardware, the operating system, installing, the USB SSD, time sync, the template unit, logs, the memory and CPU targets, and moving the runs over from the Mac. Every Pi performance number below is an estimate from the development Mac and published benchmarks until it is measured on the target Pi, as [section 2.4 of the plan](../PLAN.md#24-how-many-bots-fit-on-a-raspberry-pi-4-and-on-this-laptop-question-4) explains.

## Hardware

* **A Raspberry Pi 4 Model B with 2 GB of memory or more.** One run of 20 learners and their twins is estimated at about 110 MB at its peak, so six runs fit in 2 GB, and 4 GB leaves room for tools.
* **The official 5.1 V, 3 A USB C power supply.** A weak supply causes undervoltage throttling and can drop a USB drive, and the SSD draws its power from the Pi.
* **A USB SSD** in a USB 3 enclosure, plugged into one of the blue USB 3 ports. The next sections explain why.
* **A case with a heatsink or a fan.** A Pi 4 that runs hot slows its own clock, and a throttled Pi runs noticeably slower.
* **Wired Ethernet.** Every drop of the network is a feed gap, and each gap discards up to one horizon of pending labels. Wi Fi works, but a cable removes one cause of gaps.

The memory and CPU estimates behind these choices are in [section 2.4 of the plan](../PLAN.md#24-how-many-bots-fit-on-a-raspberry-pi-4-and-on-this-laptop-question-4).

## Operating system and Python 3.11

Use Raspberry Pi OS Lite, 64 bit, written with Raspberry Pi Imager straight to the SSD. In the Imager settings set the host name, a user, and an SSH key, and leave the desktop out. The examples below assume the host name `btcpi.local` and a user named `btc`.

Two images work. The current Trixie image ships Python 3.13, and the Legacy image based on Bookworm ships Python 3.11. The project needs Python 3.11 or newer and its test suite runs on both. `pyproject.toml` installs numpy 2.4.6 on Python 3.11 and numpy 2.5.3 on 3.12 and newer, because numpy 2.5 dropped 3.11.

For the runs that move from the Mac, the Legacy image is the better choice. The live runs on the Mac use Python 3.11 in `.venv311`, so the Legacy image resumes them with the same Python and the same numpy version, and the move changes as little as possible. A fresh run that starts on the Pi can use either image. The supported platforms are listed in [section 19 of the plan](../PLAN.md#19-packaging-and-platform), and the images are described in the [Raspberry Pi OS documentation](https://www.raspberrypi.com/documentation/computers/os.html).

## Installing

Update the system and install the few packages the project needs. `sqlite3` reads the databases by hand, and `git` checks the code before runs move over from the Mac.

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv sqlite3 rsync git
```

Copy the project from the Mac without its virtual environments, its data or its local `config.toml`. Copying the folder brings the commit the Mac has checked out, with its history, and needs no access to the repository from the Pi.

```bash
# on the Mac
rsync -a --exclude '/.venv*' --exclude /data --exclude /config.toml ~/code/btc_survivor/ btc@btcpi.local:btc_survivor/
```

Then create the virtual environment on the Pi, install the package with its test extra, and run the whole suite. A plain install rather than an editable one means a later change to the source folder cannot reach a running experiment until the package is installed again. matplotlib is never installed on the Pi.

```bash
cd ~/btc_survivor
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install ".[test]"
.venv/bin/pytest -q
```

A run that starts fresh on the Pi creates its database on its first start. It can also be created beforehand with `.venv/bin/btc_survivor --config configs/run_a.toml init-db`, which prints the path and so confirms where the config points. A run that moves from the Mac brings its database with it, as the last section explains. The commands are listed in the [README](../README.md) and in [operations.md](operations.md).

## Why a USB SSD

Each run writes to SQLite all day. Feed events and the runtime status row are committed every minute, and the buffered rows and a checkpoint every 300 seconds, so one run of 20 learners with their twins grows by roughly 700 MB a year. The log files and the system journal add their own writes, and five runs write five times as much. SD cards wear out under many small writes and are more easily corrupted by a power cut in the middle of one. A slow card also hurts the runs directly, because every database write happens on the event loop thread, and while a write blocks, the feed is not read.

An SSD on USB 3 removes both problems. Pi 4 boards shipped in recent years boot from USB out of the box. An older board may need its bootloader updated with `sudo rpi-eeprom-update -a` and USB boot chosen in `sudo raspi-config` under Advanced Options and Boot Order. Booting from the SSD is simpler than keeping the system on a card and only the data on the SSD, and it keeps the journal off the card as well.

Write durations are visible in the logs. A database write that takes longer than one second is logged as `slow database write`, and a minute step longer than 10 seconds as `slow minute step`. Neither should appear on an SSD. The storage design is in [section 12 of the plan](../PLAN.md#12-storage-schema) and SQLite's write ahead log is described in [the SQLite documentation](https://www.sqlite.org/wal.html).

## Time sync

A Pi 4 has no battery backed clock. At boot it restores the time saved at the last shutdown and only learns the real time when `systemd-timesyncd` reaches an NTP server. Until then its clock can be minutes or days behind. Every event in a run is stamped with the local UTC time on arrival, so a run must not start before the clock is right.

The unit starts after `time-sync.target`, but by default that target does not wait for a real sync. Enabling one service makes it wait.

```bash
sudo systemctl enable systemd-time-wait-sync.service
timedatectl status
```

`timedatectl status` should report that the system clock is synchronized. With the service enabled, a boot without a network holds the runs back until the first sync, which costs nothing, because they could not read the feed anyway.

The runs also guard themselves. The feed health monitor compares the exchange's message timestamps with local time, and when the median gap over the last 50 messages passes 2 seconds, or 5 in run D2, it writes a `clock_skew` feed event and holds the feed out of Healthy until a clean minute, so nothing trades or learns on a wrong clock. On the Mac this event has also fired when messages arrived late, so on a Pi whose clock is synchronized a `clock_skew` event points at slow delivery on the network rather than at the clock. The guard is described in [section 7 of the plan](../PLAN.md#7-feed) and the service in the [systemd documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd-time-wait-sync.service.html).

## The systemd template unit

`deploy/btc_survivor@.service` is a template, so one file runs every run as its own service. The text after the `@` names the config, so `btc_survivor@run_a` runs `configs/run_a.toml`, `btc_survivor@run_b` runs `configs/run_b.toml`, and likewise `btc_survivor@run_c`, `btc_survivor@run_d`, `btc_survivor@run_d2` and `btc_survivor@run_e`. Each run's config points at its own data folder under `data/`.

The unit has two placeholders, `User` and `WorkingDirectory`, both written as `CHANGE_ME`, and the path in `ExecStart` uses the same folder. Fill them in while installing the unit, then start the runs.

```bash
sed 's/CHANGE_ME/btc/g' ~/btc_survivor/deploy/btc_survivor@.service | sudo tee /etc/systemd/system/btc_survivor@.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now btc_survivor@run_a btc_survivor@run_b btc_survivor@run_c btc_survivor@run_d btc_survivor@run_d2 btc_survivor@run_e
systemctl status 'btc_survivor@*'
```

What each setting does.

* `ExecStart` runs `btc_survivor --config configs/%i.toml run` from the virtual environment, with `%i` replaced by the instance name.
* `Restart=always` and `RestartSec=5` bring a run back five seconds after a crash or a kill. A restart is never a death. The run resumes from its newest checkpoint, charges no rent for the downtime, and flattens every open position if the gap was longer than 10 minutes.
* `KillSignal=SIGTERM` and `TimeoutStopSec=60` give a run up to a minute to stop cleanly. On SIGTERM the runner finishes the current minute step, waits up to 1.3 seconds for pending fills, writes a final checkpoint and exits with code 0.
* `After=network-online.target time-sync.target` and `Wants=network-online.target` start the runs only once the network is up and the clock is synchronized.
* `Environment=PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1` sends log lines to the journal at once and stops numpy's OpenBLAS from starting a thread per core, which only adds overhead for a network this small and would make five runs compete for the cores.

To stop one run, use `sudo systemctl stop btc_survivor@run_a`. To stop it from starting at boot, use `sudo systemctl disable btc_survivor@run_a`. Shutdown and resume are described in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown) and the unit settings in the [systemd service documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html).

## Logs

Each run logs to standard error, which systemd keeps in the journal, and to its own file, `data/run_a/logs/btc_survivor.log` for run A, rotated at UTC midnight and kept for 14 days. All timestamps are UTC.

```bash
journalctl -u btc_survivor@run_a -f                              # follow run A
journalctl -u 'btc_survivor@*' --since "1 hour ago"              # every run, last hour
journalctl -u btc_survivor@run_a | grep -E ' (WARNING|ERROR) '   # warnings and errors only
journalctl -u btc_survivor@run_a --since today | grep heartbeat | tail -5
```

Once a minute each run logs a heartbeat line with the feed state, the number of living learners, their mean, lowest and highest equity, deaths so far, current and peak memory, and how late the one second tick ran. Lines to look for are the warnings `slow database write` and `slow minute step`, `feed event` lines such as `clock_skew`, `stale`, `disconnected`, `connected` and `resume`, the `reconnecting in` line logged each time a connection ends, and `invariant violation`, which is logged at ERROR. A reconnect is only a log line, and the feed events around it are `disconnected` and `connected`. systemd stores every line of standard error at the same priority, so filter by the level word in the line rather than with `journalctl -p`. The feed events are also in the database.

```bash
sqlite3 -readonly ~/btc_survivor/data/run_a/btc_survivor.db \
  "SELECT ts, kind, detail FROM feed_events ORDER BY id DESC LIMIT 20"
```

The file logs already keep two weeks, so a modest journal limit such as `SystemMaxUse=200M` in `/etc/systemd/journald.conf` is enough. The dashboard listens on 127.0.0.1 only, so on a headless Pi it is not reachable from other machines, and `.venv/bin/btc_survivor --config configs/run_a.toml status` over SSH shows the same numbers as text. The commands are described in [operations.md](operations.md) and the journal in the [journalctl documentation](https://www.freedesktop.org/software/systemd/man/latest/journalctl.html).

## Memory and CPU targets

The target for each run is 20 learners, with their twins, under 150 MB of resident memory and under 25 percent of one core on average, in a single process. Each run has its own target, so six runs together use about six times the memory and feed CPU of one, and they share the Pi's four cores.

The estimates leave room. Memory is estimated at about 110 MB at the peak, about half of it the baseline and half a brief spike while the 4.5 MB order book snapshot is parsed after a reconnect. CPU is estimated at 1 to 4 percent of one core at the measured quiet rate of 17 messages a second, and at about 6 percent in the central case and 24 percent in the pessimistic case at an assumed busy rate of 100 messages a second. The bots themselves cost only about 9 milliseconds a minute in total, and handling the feed is nearly all of the load.

Measure both on the Pi. The heartbeat line gives current and peak memory, read from `/proc/self/statm` and the kernel's peak count, and the same two numbers are stored in the `runtime_status` table. `ps` gives the average CPU since the process started, which is the number the target refers to.

```bash
pid=$(systemctl show -p MainPID --value btc_survivor@run_a)
ps -o pid,rss,pcpu,etime,args -p "$pid"     # rss in KiB, pcpu averaged since start
vcgencmd get_config arm_freq                # top clock in MHz, 1500 on older boards, 1800 on newer ones
vcgencmd measure_clock arm                  # clock right now, lower when idle
vcgencmd measure_temp
vcgencmd get_throttled                      # throttled=0x0 means no throttling since boot
```

The Raspberry Pi kernel has long shipped with the memory cgroup turned off. If `memory` is missing from `/sys/fs/cgroup/cgroup.controllers`, then `MemoryMax` has no effect and `systemctl status` shows no memory figure, so the heartbeat and `ps` are the measures to trust. If the peak comes too close to 150 MB after many reconnects, the plan's fixes are a Kraken adapter with its 2 KB snapshot or a streaming parse, and either change waits for the owner's approval. The figures come from [section 2.4 of the plan](../PLAN.md#24-how-many-bots-fit-on-a-raspberry-pi-4-and-on-this-laptop-question-4) and the risks from [section 20](../PLAN.md#20-honest-limits-and-risks).

## Moving runs from the Mac

Runs A to D started on 11 October 2026 on a MacBook, and the hypothetical run E on 12 October. A Mac with its lid closed on battery sleeps even with a keep awake tool such as Amphetamine, and a sleeping Mac freezes every run until it wakes, so on the Mac the runs need mains power. The Pi does not sleep, which is the main reason to move them. A run is just its config and its data folder, so moving it means stopping it cleanly, copying the folder and starting it on the Pi.

1. Install and test the project on the Pi first, as above, without starting any service.
2. Make sure the Pi holds the same trading code the runs started with. Runs A to C launched from commit `174718f`, run D from `ef648f5` with its config not yet committed, and run E from the commit recorded for it, as the run header of [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md#run-header) records. rsync copies the working tree rather than the last commit, so first check that nothing in `src` or `configs` is uncommitted, then list what changed since each launch outside the reporting code, with one more diff of the same form from run E's launch commit.

   ```bash
   # on the Pi, from ~/btc_survivor
   git status --short src configs
   git diff --stat 174718f HEAD -- src ':(exclude)src/btc_survivor/cli' ':(exclude)src/btc_survivor/dashboard.*'
   git diff --stat ef648f5 HEAD -- src ':(exclude)src/btc_survivor/cli' ':(exclude)src/btc_survivor/dashboard.*'
   ```

   `git status` should print nothing. Changes to the reporting code, the `cli` folder, the dashboard and the reporting commands in `__main__.py`, are expected and harmless, because none of them changes how a run trades or learns. So the first diff should list only `src/btc_survivor/__main__.py`, where `git diff 174718f HEAD -- src/btc_survivor/__main__.py` shows the new commands and a longer list of errors that `main` prints as one line instead of a traceback, and the second diff, like the one from run E's launch commit, should print nothing. Any other file in either list is a change to the trading path, and the move waits until that change is understood. Comparing `git rev-parse HEAD` or file hashes between the Mac and the Pi proves nothing right after an rsync, because the copy made them identical. The configs are checked against the text each run stored at launch in step 5.
3. Stop the runs on the Mac with SIGTERM, using the stop command in [operations.md](operations.md#stopping-runs), `pkill -TERM -f 'configs/run_a.toml run'` and the same for runs B, C, D and E, and wait until `pgrep -fl btc_survivor` no longer lists them. Ctrl+C does the same for a run started in a foreground terminal. A clean stop writes a final checkpoint and logs `stopped cleanly`. Stop the Mac dashboard too, since from then on it would only show the old copies.
4. Copy the five data folders, including any `-wal` or `-shm` file next to a database.

   ```bash
   # on the Mac, from ~/code/btc_survivor
   rsync -a data/run_a data/run_b data/run_c data/run_d data/run_d2 data/run_e btc@btcpi.local:btc_survivor/data/
   ```

5. On the Pi, check each database with `.venv/bin/btc_survivor --config configs/run_a.toml init-db`, which reports that it matches the schema, and look at it with `status`. Then compare each config with the text the run stored in its `meta` table at launch. They must match byte for byte.

   ```bash
   # on the Pi, from ~/btc_survivor
   for run in run_a run_b run_c run_d run_d2 run_e; do
     sqlite3 -readonly data/$run/btc_survivor.db "SELECT rtrim(value, char(10)) FROM meta WHERE key = 'config_toml'" \
       | diff - configs/$run.toml && echo "$run config matches"
   done
   ```

   The `rtrim` drops the final newline of the stored text, because `sqlite3` adds one when it prints. A run refuses to resume when its master seed, population size or twin setting differs, and no other setting may change during a run either.
6. Make sure the clock is synchronized, then install and start the units as in the template unit section above.
7. Watch the first minutes in the journal. Each run should log a `resume` feed event that names the restored checkpoint and then reach Healthy.

What the move costs. The time between the stop on the Mac and the first healthy minute on the Pi is a gap. No rent is charged for it, pending labels are discarded and pending orders rejected as after any restart, and because a move takes longer than 10 minutes every open position is closed with exit reason `restart_flatten` at the first healthy minute, paying its normal closing fee. The success criteria count healthy days, so the final day arrives later by the length of the gap. Record the move in the hardware row and the interruptions table of [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md).

Two more points. Never start the same run on both machines, because two processes writing two copies of one run would split its history, so keep the Mac copy only as a backup and never start it again. And numpy on the Pi uses OpenBLAS where numpy on a recent Mac uses Apple's Accelerate, so the last bits of some floating point results can differ after the move. That does not matter for a live run, which is never replayed bit for bit. Resume is described in [section 13 of the plan](../PLAN.md#13-checkpoints-resume-and-shutdown).

## Checklist for the first 24 hours on the Pi

Before trusting the Pi with the runs, check these over the first day.

* Memory at the end of 24 hours is no more than 5 MB above memory at the end of the first hour, read from the heartbeat lines.
* Before the runs move, `python scripts/soak_test.py` runs once on the Pi. Its memory reading comes from `/proc`, so its verdict is `passed` or `failed`, and its peak shows how much room the largest snapshots leave under 150 MB, as [operations.md](operations.md#the-soak-test) explains.
* Average CPU per run stays under 25 percent of one core, and `vcgencmd get_throttled` still prints `throttled=0x0`.
* No `slow database write`, `slow minute step` or `invariant_violation` appears in the journal.
* Pull the network cable for about a minute and plug it back in. The feed goes stale, reconnects and returns to Healthy. The feed events include `stale`, `disconnected`, `connected`, `snapshot`, `recovering` and `healthy`, the journal shows the `reconnecting in` lines, and no rent is charged for the outage.
* Kill one run with `sudo systemctl kill -s SIGKILL btc_survivor@run_a`. systemd restarts it after five seconds, a `resume` feed event appears, and its books still balance.

These checks follow [section 19 of the plan](../PLAN.md#19-packaging-and-platform) and the soak criteria in [section 18](../PLAN.md#18-build-phases).
