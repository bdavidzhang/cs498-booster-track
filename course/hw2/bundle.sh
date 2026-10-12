REINFORCE_RUN="2026-09-26_15-33-52_reinforce"
T100_RUN="2026-09-28_10-05-36_ppo_T100"
T24_RUN="2026-09-28_11-15-47_T24"

T24_LOG="PPO_T24_LOG.out"
T100_LOG="PPO_T100_LOG.out"
REINFORCE_LOG="REINFORCE_LOG.out"

tar czf hw2/hw2_runs.tar.gz \
  logs/rsl_rl/hw2_reinforce/$REINFORCE_RUN \
  logs/rsl_rl/hw2_ppo/$T100_RUN \
  logs/rsl_rl/hw2_ppo/$T24_RUN \
  logs/$T24_LOG logs/$T100_LOG logs/$REINFORCE_LOG \
  hw2/comparison.png