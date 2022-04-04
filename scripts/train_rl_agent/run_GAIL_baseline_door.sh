python -m tools.run_rl configs/gail/gail_baseline.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDoor-v0" \
--work-dir=./work_dirs/gail_baseline_door --num-gpus=3 --clean-up