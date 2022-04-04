python -m tools.run_rl configs/sac/sac_transformer.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDoor-v0" \
--work-dir=./work_dirs/sac_transformer_door/ --num-gpus=3 --clean-up